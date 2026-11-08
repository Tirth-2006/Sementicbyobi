"""The installable package must not depend on repository scripts."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE_FILES = sorted((SRC / "uncertainty").rglob("*.py"))

# Modules that need torch/transformers/datasets at import time are covered by
# the clean-environment wheel test in CI; the rest must import everywhere.
LIGHT_MODULES = [
    "uncertainty",
    "uncertainty.config",
    "uncertainty.models",
    "uncertainty.semantic.clustering",
    "uncertainty.semantic.semantic_entropy",
    "uncertainty.evaluation.correctness",
    "uncertainty.evaluation.metrics",
    "uncertainty.evaluation.predictive_probes",
    "uncertainty.evaluation.selective_prediction",
    "uncertainty.hidden_states.dispersion",
    "uncertainty.hidden_states.extraction",
    "uncertainty.token_uncertainty.predictive_entropy",
    "uncertainty.token_uncertainty.teacher_forcing",
    "uncertainty.utils.alignment",
    "uncertainty.utils.provenance",
    "uncertainty.utils.validation",
]


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_package_files_were_found():
    assert len(PACKAGE_FILES) > 20


@pytest.mark.parametrize("path", PACKAGE_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_package_never_imports_scripts_or_src_alias(path):
    offenders = {
        name
        for name in imported_modules(path)
        if name.split(".")[0] in {"scripts", "src", "evaluate_low_temp_correctness"}
    }
    assert not offenders, f"{path.name} imports repository-only modules: {offenders}"


def test_no_user_specific_absolute_paths_in_source():
    root = SRC.parents[0]
    offenders = []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "C:\\Users" in text or "OneDrive" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, offenders


def test_scripts_do_not_use_cwd_as_model_directory():
    root = SRC.parents[0]
    offenders = [
        str(p.relative_to(root))
        for p in (root / "scripts").rglob("*.py")
        if 'MODEL_DIR = Path(".")' in p.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_modules_import_outside_repo_root(module, tmp_path):
    """Import from a foreign cwd with only ``src`` on the path (no repo root)."""
    pytest.importorskip("numpy")
    code = (
        "import sys; sys.path[:] = [p for p in sys.path if p not in ('', '.')];"
        f"sys.path.insert(0, {str(SRC)!r});"
        f"import importlib; m = importlib.import_module({module!r});"
        "assert 'scripts' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
