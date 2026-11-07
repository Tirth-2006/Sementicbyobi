"""Dataset loading utilities."""
import random
from typing import List, Dict, Tuple
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriviaQALoader:
    """Load and process TriviaQA dataset."""

    def __init__(self, seed: int = 42):
        """Initialize loader.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def load(self):
        """Load TriviaQA dataset from HuggingFace."""
        logger.info("Loading TriviaQA dataset...")

        # Load dataset - unfiltered.nocontext provides questions without context
        # Use full namespace to avoid deprecation issues
        dataset = load_dataset('mandarjoshi/trivia_qa', 'unfiltered.nocontext')

        # Split into train and validation
        # TriviaQA has 'train' and 'validation' splits
        self.train_data = dataset['train']
        self.val_data = dataset['validation']

        logger.info(f"Loaded {len(self.train_data)} training examples")
        logger.info(f"Loaded {len(self.val_data)} validation examples")

        return self

    def _format_example(self, example) -> Dict:
        """Format a single example to consistent structure.

        Args:
            example: Raw dataset example

        Returns:
            Formatted example dictionary
        """
        # TriviaQA structure: question, question_id, answer (dict with value, aliases)
        return {
            'id': example['question_id'],
            'question': example['question'],
            'answers': {
                'text': [example['answer']['value']] + example['answer'].get('aliases', []),
                'answer_start': [0] * (1 + len(example['answer'].get('aliases', [])))
            },
            'context': None  # TriviaQA unfiltered.nocontext has no context
        }

    def get_train_examples(self, num_examples: int = None) -> List[Dict]:
        """Get training examples.

        Args:
            num_examples: Number of examples to return (None = all)

        Returns:
            List of formatted examples
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        if num_examples is None:
            indices = range(len(self.train_data))
        else:
            indices = random.sample(range(len(self.train_data)),
                                   min(num_examples, len(self.train_data)))

        return [self._format_example(self.train_data[i]) for i in indices]

    def get_validation_examples(self, num_examples: int = None) -> List[Dict]:
        """Get validation examples.

        Args:
            num_examples: Number of examples to return (None = all)

        Returns:
            List of formatted examples
        """
        if self.val_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        if num_examples is None:
            indices = range(len(self.val_data))
        else:
            indices = random.sample(range(len(self.val_data)),
                                   min(num_examples, len(self.val_data)))

        return [self._format_example(self.val_data[i]) for i in indices]

    def select_few_shot_examples(self, num_examples: int = 5) -> Tuple[List[int], List[Dict]]:
        """Select examples for few-shot prompting.

        Args:
            num_examples: Number of few-shot examples

        Returns:
            Tuple of (indices, examples)
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Select random indices from training set
        indices = random.sample(range(len(self.train_data)), num_examples)
        examples = [self._format_example(self.train_data[i]) for i in indices]

        logger.info(f"Selected {num_examples} few-shot examples from indices: {indices}")

        return indices, examples


def load_dataset_by_name(dataset_name: str, seed: int = 42):
    """Load dataset by name.

    Args:
        dataset_name: Name of dataset ('trivia_qa', 'squad', etc.)
        seed: Random seed

    Returns:
        Dataset loader instance
    """
    if dataset_name == "trivia_qa":
        return TriviaQALoader(seed=seed).load()
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not yet implemented")
