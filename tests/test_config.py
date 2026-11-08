import dataclasses

import pytest

from uncertainty.config import (
    MODEL_DIR_ENV_VAR,
    ExperimentConfig,
    ModelConfig,
    get_default_config,
    resolve_model_dir,
)


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "Qwen3-1.7B"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


def test_no_hardcoded_default_model_directory():
    assert ModelConfig().model_dir is None


def test_unset_model_dir_fails_with_clear_message(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=MODEL_DIR_ENV_VAR):
        resolve_model_dir()


def test_env_var_is_used(monkeypatch, model_dir):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(model_dir))
    assert resolve_model_dir() == model_dir


def test_explicit_argument_beats_env_var(monkeypatch, model_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text("{}")
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(model_dir))
    assert resolve_model_dir(other) == other


def test_missing_directory_and_missing_config_fail(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_dir(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_model_dir(empty)


def test_model_config_resolves_through_env(monkeypatch, model_dir):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(model_dir))
    assert ModelConfig().resolved_model_dir() == model_dir


def test_default_config_values_match_documented_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ExperimentConfig creates its output dir
    config = get_default_config()
    assert config.generation.temperature_high == 1.0
    assert config.generation.temperature_low == 0.1
    assert config.generation.num_generations == 10
    assert config.data.num_questions == 100
    assert config.data.num_few_shot == 5


def test_config_round_trip_has_no_absolute_user_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = get_default_config()
    path = tmp_path / "config.json"
    config.save(str(path))
    loaded = ExperimentConfig.load(str(path))
    assert dataclasses.asdict(loaded.model) == dataclasses.asdict(config.model)
    text = path.read_text()
    assert "Users" not in text and "C:\\\\" not in text
