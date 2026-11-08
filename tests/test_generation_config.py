import pytest

from uncertainty.config import GenerationConfig
from uncertainty.generation.sampling import resolve_sampling_kwargs


def test_experiment_defaults_sample_at_both_temperatures():
    config = GenerationConfig()
    for temperature in (config.temperature_low, config.temperature_high):
        kwargs = resolve_sampling_kwargs(
            temperature, config.top_p, config.top_k, config.do_sample
        )
        assert kwargs == {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
        }


def test_do_sample_false_forces_greedy_and_drops_sampling_args():
    kwargs = resolve_sampling_kwargs(1.0, 0.8, 20, do_sample=False)
    assert kwargs == {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


def test_zero_temperature_is_greedy_even_if_do_sample_true():
    assert resolve_sampling_kwargs(0.0, 0.8, 20, do_sample=True)["do_sample"] is False


def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        resolve_sampling_kwargs(-0.1, 0.8, 20)


def test_enable_thinking_true_is_rejected_because_it_is_not_implemented():
    assert GenerationConfig().enable_thinking is False
    with pytest.raises(ValueError, match="not supported"):
        GenerationConfig(enable_thinking=True)


def test_saved_configs_with_both_fields_still_load():
    config = GenerationConfig(**{"do_sample": True, "enable_thinking": False})
    assert config.do_sample is True


def test_generator_does_not_accept_or_ignore_enable_thinking():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "uncertainty" / "generation" / "generator.py"
    ).read_text(encoding="utf-8")
    assert "enable_thinking" not in source
    assert "do_sample=gen_config.do_sample" in source
