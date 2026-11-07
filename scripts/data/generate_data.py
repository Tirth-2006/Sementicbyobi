"""Main script for generating semantic entropy dataset."""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from uncertainty.config import get_default_config, ExperimentConfig
from uncertainty.data.data_loader import load_dataset_by_name
from uncertainty.data.prompt_builder import create_prompt_builder
from uncertainty.generation.generator import QwenGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_dataset(config: ExperimentConfig):
    """Generate responses for semantic entropy analysis.

    Args:
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("Starting Semantic Entropy Data Generation")
    logger.info("=" * 80)

    # Load dataset
    logger.info(f"Loading dataset: {config.data.dataset_name}")
    dataset_loader = load_dataset_by_name(
        config.data.dataset_name,
        seed=config.data.random_seed
    )

    # Select few-shot examples
    logger.info(f"Selecting {config.data.num_few_shot} few-shot examples")
    few_shot_indices, few_shot_examples = dataset_loader.select_few_shot_examples(
        num_examples=config.data.num_few_shot
    )

    # Build few-shot prompt
    logger.info("Building few-shot prompt")
    prompt_builder = create_prompt_builder(brief_type='default')
    few_shot_prompt = prompt_builder.build_few_shot_prompt(few_shot_examples)

    logger.info("Few-shot prompt:")
    logger.info("-" * 80)
    logger.info(few_shot_prompt)
    logger.info("-" * 80)

    # Load model
    logger.info("Loading model...")
    generator = QwenGenerator(config.model)
    logger.info(f"GPU memory allocated: {generator.get_gpu_memory_mb():.1f} MiB")

    # Get validation questions
    logger.info(f"Loading {config.data.num_questions} validation questions")
    validation_examples = dataset_loader.get_validation_examples(
        num_examples=config.data.num_questions
    )

    # Generate responses
    logger.info("=" * 80)
    logger.info("Starting response generation")
    logger.info("=" * 80)

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'few_shot_indices': few_shot_indices,
            'few_shot_prompt': few_shot_prompt,
            'num_questions': len(validation_examples)
        },
        'generations': {}
    }

    for idx, example in enumerate(tqdm(validation_examples, desc="Generating responses")):
        question_id = example['id']
        question = example['question']
        context = example.get('context')
        reference_answers = example['answers']['text']

        logger.info(f"\n[{idx+1}/{len(validation_examples)}] Question ID: {question_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Reference answers: {reference_answers}")

        # Build complete prompt for this question
        complete_prompt = prompt_builder.build_query_prompt(
            few_shot_prompt, question, context
        )

        # Generate responses
        try:
            generations = generator.generate_multiple(
                prompt=complete_prompt,
                gen_config=config.generation
            )

            # Store results
            results['generations'][question_id] = {
                'question': question,
                'context': context,
                'reference_answers': reference_answers,
                'low_temp_generation': generations['low_temp'],
                'high_temp_generations': generations['high_temp'],
            }

            logger.info(f"Low-temp answer: {generations['low_temp']['response']}")
            logger.info(f"Generated {len(generations['high_temp']['responses'])} high-temp answers")

        except Exception as e:
            logger.error(f"Error generating for question {question_id}: {e}")
            results['generations'][question_id] = {
                'question': question,
                'context': context,
                'reference_answers': reference_answers,
                'error': str(e)
            }

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{config.data.dataset_name}_qwen3_{timestamp}.json"

    logger.info("=" * 80)
    logger.info(f"Saving results to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save config separately
    config_file = output_dir / f"{config.data.dataset_name}_qwen3_{timestamp}_config.json"
    config.save(str(config_file))

    logger.info(f"Configuration saved to: {config_file}")
    logger.info("=" * 80)
    logger.info("Generation complete!")
    logger.info(f"Total questions processed: {len(results['generations'])}")
    logger.info(f"Final GPU memory: {generator.get_gpu_memory_mb():.1f} MiB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate semantic entropy dataset using Qwen3-1.7B"
    )

    parser.add_argument(
        "--dataset", type=str, default="trivia_qa",
        choices=["trivia_qa"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_questions", type=int, default=100,
        help="Number of questions to process"
    )
    parser.add_argument(
        "--num_few_shot", type=int, default=5,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--num_generations", type=int, default=10,
        help="Number of high-temperature generations per question"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for high-temperature sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/generated",
        help="Output directory"
    )

    args = parser.parse_args()

    # Create configuration
    config = get_default_config()

    # Override with command-line arguments
    config.data.dataset_name = args.dataset
    config.data.num_questions = args.num_questions
    config.data.num_few_shot = args.num_few_shot
    config.data.random_seed = args.seed
    config.generation.num_generations = args.num_generations
    config.generation.temperature_high = args.temperature
    config.output_dir = args.output_dir

    # Run generation
    generate_dataset(config)


if __name__ == "__main__":
    main()
