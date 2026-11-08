"""Model wrapper and generation utilities."""
from typing import Any, Dict
import logging

import torch

from ..config import ModelConfig, GenerationConfig
from ..models import load_quantized_causal_lm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Stop sequences to prevent model from continuing few-shot pattern
# From Farquhar et al. semantic uncertainty paper
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


class QwenGenerator:
    """Wrapper for Qwen3-1.7B model with generation utilities."""

    def __init__(self, model_config: ModelConfig):
        """Initialize model.

        Args:
            model_config: Model configuration
        """
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check PyTorch installation.")

        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Loading {self.config.model_name} (4-bit NF4) from local files...")

        # Single shared loader: same quantization settings for every script.
        self.tokenizer, self.model, _ = load_quantized_causal_lm(
            self.config.resolved_model_dir(),
            device_map=self.config.device_map,
        )
        logger.info("Model loaded successfully.")

    def generate_single(self,
                       prompt: str,
                       temperature: float,
                       top_p: float = 0.8,
                       top_k: int = 20,
                       max_new_tokens: int = 64,
                       enable_thinking: bool = False) -> str:
        """Generate a single response.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            enable_thinking: Enable Qwen3 thinking mode

        Returns:
            Generated answer text (cleaned)
        """
        # DON'T use chat template for few-shot prompts - use raw text
        # The few-shot prompt is already formatted correctly
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        # Determine sampling mode
        do_sample = temperature > 0.0

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (skip prompt)
        answer_ids = generated[0, prompt_len:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Clean the answer: stop at any stop sequence
        cleaned_answer = self._clean_answer(answer)

        return cleaned_answer

    def _clean_answer(self, answer: str) -> str:
        """Clean answer by removing stop sequences and extra content.

        Args:
            answer: Raw generated answer

        Returns:
            Cleaned answer text
        """
        # Stop at any of the stop sequences
        for stop_seq in STOP_SEQUENCES:
            if stop_seq in answer:
                answer = answer.split(stop_seq)[0]
                break

        # Remove leading/trailing whitespace
        answer = answer.strip()

        return answer

    def generate_multiple(self,
                         prompt: str,
                         gen_config: GenerationConfig) -> Dict[str, Any]:
        """Generate multiple responses for semantic entropy estimation.

        Args:
            prompt: Input prompt text
            gen_config: Generation configuration

        Returns:
            Dictionary with 'low_temp' and 'high_temp' generations
        """
        results = {}

        # Generate low-temperature response for accuracy baseline
        logger.debug("Generating low-temperature response...")
        low_temp_answer = self.generate_single(
            prompt=prompt,
            temperature=gen_config.temperature_low,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            max_new_tokens=gen_config.max_new_tokens,
            enable_thinking=gen_config.enable_thinking
        )

        results['low_temp'] = {
            'response': low_temp_answer,
            'temperature': gen_config.temperature_low
        }

        logger.debug(f"Low-temp answer: {low_temp_answer}")

        # Generate high-temperature responses for semantic entropy
        logger.debug(f"Generating {gen_config.num_generations} high-temperature responses...")
        high_temp_answers = []

        for i in range(gen_config.num_generations):
            answer = self.generate_single(
                prompt=prompt,
                temperature=gen_config.temperature_high,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                max_new_tokens=gen_config.max_new_tokens,
                enable_thinking=gen_config.enable_thinking
            )
            high_temp_answers.append({
                'response': answer,
                'index': i
            })
            logger.debug(f"  [{i+1}] {answer}")

        results['high_temp'] = {
            'responses': high_temp_answers,
            'temperature': gen_config.temperature_high,
            'num_generations': gen_config.num_generations
        }

        return results

    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory allocated in MiB."""
        return torch.cuda.memory_allocated(0) / (1024 ** 2)
