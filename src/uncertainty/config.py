"""Configuration management for semantic entropy experiments."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen3-1.7B"
    model_dir: str = r"C:\Users\TIRTH\OneDrive\Desktop\Hallucination"
    quantization: str = "4bit"
    device_map: str = "auto"


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

    # Sampling
    do_sample: bool = True
    enable_thinking: bool = False  # Disable Qwen3 thinking mode


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
