from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DIR = r"C:\Users\TIRTH\OneDrive\Desktop\Hallucination"

def main():
    if not (Path(MODEL_DIR) / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {MODEL_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the PyTorch installation.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer from local files...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading Qwen3-1.7B in 4-bit mode...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=True
    )

    model.eval()
    print("Model loaded successfully.")

    messages = [{
        "role": "user",
        "content": "What is the capital of France? Answer in one short sentence."
    }]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_len = inputs["input_ids"].shape[1]

    print("Generating answer...")

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        full_pass = model(
            input_ids=generated,
            attention_mask=torch.ones_like(generated),
            output_hidden_states=True,
            use_cache=False
        )

    seq_len = generated.shape[1]
    answer_ids = generated[0, prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    print("\n--- TEST RESULTS ---")
    print("Answer:", answer.strip())

    # Logits at each position predict the next token.
    answer_logits = full_pass.logits[0, prompt_len - 1:seq_len - 1].float()
    log_probs = torch.log_softmax(answer_logits, dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)

    print("Generated tokens:", answer_ids.numel())
    print("Mean token entropy:", round(entropies.mean().item(), 4))

    # Last layer's representations for the answer tokens.
    answer_states = full_pass.hidden_states[-1][
        0, prompt_len:seq_len, :
    ].float()

    pooled_hidden = answer_states.mean(dim=0)

    print("Hidden-state layers returned:", len(full_pass.hidden_states))
    print("Final hidden-state tensor shape:", tuple(full_pass.hidden_states[-1].shape))
    print("Pooled answer vector shape:", tuple(pooled_hidden.shape))
    print("GPU memory allocated (MiB):",
          round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1))
    print("\nTest completed.")

if __name__ == "__main__":
    main()
