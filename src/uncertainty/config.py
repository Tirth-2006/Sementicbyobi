"""Configuration management for semantic entropy experiments."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json
import os

MODEL_DIR_ENV_VAR = "UNCERTAINTY_MODEL_DIR"


def resolve_model_dir(model_dir: Optional[str | os.PathLike] = None) -> Path:
    """Return a validated local model directory.

    Resolution order: the explicit ``model_dir`` argument, then the
    ``UNCERTAINTY_MODEL_DIR`` environment variable. There is deliberately no
    machine-specific default. A clear error is raised when the location is
    unset or does not look like a Hugging Face model directory.
    """
    candidate = model_dir if model_dir else os.environ.get(MODEL_DIR_ENV_VAR)
    if not candidate:
        raise RuntimeError(
            "No model directory configured. Set the "
            f"{MODEL_DIR_ENV_VAR} environment variable (or pass model_dir) "
            "to the local Qwen3-1.7B directory containing config.json."
        )
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model directory {path} does not contain config.json; "
            f"check {MODEL_DIR_ENV_VAR}."
        )
    return path


@dataclass
class ModelConfig:
    """Model configuration.

    ``model_dir`` is optional here; call :meth:`resolved_model_dir` to apply the
    environment-variable fallback and validate the location.
    """
    model_name: str = "Qwen3-1.7B"
    model_dir: Optional[str] = None
    quantization: str = "4bit"
    device_map: str = "auto"

    def resolved_model_dir(self) -> Path:
        """Return the validated model directory (argument, then env var)."""
        return resolve_model_dir(self.model_dir)


@dataclass
class GenerationConfig:
    """Generation parameters."""
    # Low-temperature generation for accuracy baseline
    temperature_low: float = 0.1

    # High-temperature sampling for semantic entropy
    temperature_high: float = 1.0
    top_p: float = 0.8
    top_k: int = 20

    # Number of generations
    num_generations: int = 10

    # Token limits
    max_new_tokens: int = 64

    # Sampling. ``do_sample`` is honoured by the generator (sampling happens only
    # when it is true AND the temperature is > 0; otherwise decoding is greedy).
    do_sample: bool = True
    # Recorded for protocol documentation only. Prompts are raw few-shot text and
    # no chat template is applied, so Qwen3's thinking toggle is never used.
    # ``True`` is not implemented and is rejected.
    enable_thinking: bool = False

    def __post_init__(self):
        if self.enable_thinking:
            raise ValueError(
                "enable_thinking=True is not supported: generation uses raw "
                "few-shot text prompts without a chat template."
            )


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "trivia_qa"
    num_questions: int = 100
    num_few_shot: int = 5
    random_seed: int = 42
    split: str = "validation"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    generation: GenerationConfig
    data: DataConfig
    output_dir: str = "data/generated"

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'model': asdict(self.model),
            'generation': asdict(self.generation),
            'data': asdict(self.data),
            'output_dir': self.output_dir
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data['model']),
            generation=GenerationConfig(**data['generation']),
            data=DataConfig(**data['data']),
            output_dir=data['output_dir']
        )


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(
        model=ModelConfig(),
        generation=GenerationConfig(),
        data=DataConfig()
    )
