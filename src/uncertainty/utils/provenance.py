"""Record checkpoint and environment provenance for reproducibility.

The original Qwen3-1.7B checkpoint revision/hash was never recorded. This
module lets anyone (including the original author, on the original machine)
fingerprint a local model directory and the Python environment so the details
can be stored next to results.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# Files that define a checkpoint + tokenizer; weights dominate the hashing time.
DEFAULT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "*.safetensors",
    "model.safetensors.index.json",
)

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_directory(
    model_dir: str | os.PathLike, patterns: tuple[str, ...] = DEFAULT_PATTERNS
) -> dict[str, dict[str, Any]]:
    """SHA-256 and size for every checkpoint/tokenizer file in ``model_dir``."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    files: dict[str, Path] = {}
    for pattern in patterns:
        for match in sorted(root.glob(pattern)):
            if match.is_file():
                files[match.name] = match
    if not files:
        raise FileNotFoundError(f"No checkpoint or tokenizer files found in {root}")
    return {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }


def collect_environment() -> dict[str, Any]:
    """Python, OS and key package versions (missing packages -> ``None``)."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def build_provenance_record(
    model_dir: str | os.PathLike, model_name: str = "Qwen3-1.7B"
) -> dict[str, Any]:
    """Provenance record to store beside experiment outputs.

    The absolute path is deliberately *not* recorded (it is machine-specific
    and may identify the user); only the directory's own name is kept. When the
    directory sits in a Hugging Face cache ``snapshots/<revision>`` folder, the
    folder name is the commit revision and is recorded as such.
    """
    root = Path(model_dir).resolve()
    revision = root.name if root.parent.name == "snapshots" else None
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_dir_name": root.name,
        "hf_snapshot_revision": revision,
        "files": hash_model_directory(root),
        "environment": collect_environment(),
    }


def write_provenance(
    model_dir: str | os.PathLike, output: str | os.PathLike
) -> dict[str, Any]:
    """Build and save the provenance record as JSON."""
    record = build_provenance_record(model_dir)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
