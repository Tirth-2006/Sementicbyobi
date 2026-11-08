"""Shared local-model loading for all model-backed scripts.

Every script that needs Qwen3 goes through this module, so the model location
comes from one place: the ``model_dir`` argument or the
``UNCERTAINTY_MODEL_DIR`` environment variable (see :mod:`uncertainty.config`).
Heavy imports (``torch``/``transformers``) happen only when a loader is called.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_tokenizer(model_dir: str | os.PathLike | None = None) -> Any:
    """Load the local tokenizer (offline)."""
    from transformers import AutoTokenizer

    from .config import resolve_model_dir

    path = resolve_model_dir(model_dir)
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def load_quantized_causal_lm(
    model_dir: str | os.PathLike | None = None,
    device_map: str | dict[str, Any] = "auto",
) -> tuple[Any, Any, Path]:
    """Load tokenizer and 4-bit NF4 (double-quant, FP16 compute) causal LM.

    Returns ``(tokenizer, model, resolved_model_dir)``; the model is in eval
    mode. Requires a CUDA-capable PyTorch/bitsandbytes installation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .config import resolve_model_dir

    path = resolve_model_dir(model_dir)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, path
