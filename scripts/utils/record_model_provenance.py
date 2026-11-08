"""CLI: fingerprint the local model checkpoint and Python environment.

Usage (from the repository root)::

    export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
    python scripts/utils/record_model_provenance.py

Writes ``results/provenance/model_provenance.json`` (small; safe to commit).
Run this on the machine that holds the checkpoint used for the reported
results, so the exact files and package versions are recorded.
"""
import argparse
from pathlib import Path

from uncertainty.config import resolve_model_dir
from uncertainty.utils.provenance import write_provenance

DEFAULT_OUTPUT = Path("results/provenance/model_provenance.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=None, help="Overrides UNCERTAINTY_MODEL_DIR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    record = write_provenance(model_dir, args.output)
    print(f"Hashed {len(record['files'])} files; saved to {args.output}")


if __name__ == "__main__":
    main()
