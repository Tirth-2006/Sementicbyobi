"""Quick test of the data generation pipeline components."""
import sys
from pathlib import Path

# Add the src layout to the import path for direct execution.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from uncertainty.config import get_default_config
from uncertainty.data.data_loader import TriviaQALoader
from uncertainty.data.prompt_builder import create_prompt_builder

print("=" * 80)
print("Testing Data Generation Pipeline Components")
print("=" * 80)

# Test 1: Configuration
print("\n1. Testing configuration...")
config = get_default_config()
print(f"   [OK] Model: {config.model.model_name}")
print(f"   [OK] Dataset: {config.data.dataset_name}")
print(f"   [OK] Num questions: {config.data.num_questions}")
print(f"   [OK] Num generations: {config.generation.num_generations}")
print(f"   [OK] Temperature (high): {config.generation.temperature_high}")
print(f"   [OK] Temperature (low): {config.generation.temperature_low}")

# Test 2: Dataset loading
print("\n2. Testing dataset loading...")
try:
    loader = TriviaQALoader(seed=42)
    loader.load()
    print(f"   [OK] Training examples: {len(loader.train_data)}")
    print(f"   [OK] Validation examples: {len(loader.val_data)}")

    # Get a sample
    val_samples = loader.get_validation_examples(num_examples=2)
    print(f"\n   Sample question:")
    print(f"   Q: {val_samples[0]['question']}")
    print(f"   A: {val_samples[0]['answers']['text']}")

except Exception as e:
    print(f"   [ERROR] Error loading dataset: {e}")
    print("\n   Note: If you see a connection error, the dataset will be")
    print("   downloaded on first run. Make sure you have internet access.")

# Test 3: Few-shot prompt construction
print("\n3. Testing few-shot prompt construction...")
try:
    few_shot_indices, few_shot_examples = loader.select_few_shot_examples(num_examples=3)
    print(f"   [OK] Selected {len(few_shot_examples)} examples")
    print(f"   [OK] Indices: {few_shot_indices}")

    prompt_builder = create_prompt_builder(brief_type='default')
    few_shot_prompt = prompt_builder.build_few_shot_prompt(few_shot_examples)
    print(f"\n   Few-shot prompt preview (first 300 chars):")
    print(f"   {few_shot_prompt[:300]}...")

    # Build a query prompt
    query = prompt_builder.build_query_prompt(
        few_shot_prompt,
        "What is the capital of France?"
    )
    print(f"\n   [OK] Complete query prompt length: {len(query)} chars")

except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 4: Model loading (optional - can be slow)
print("\n4. Testing model loading...")
print("   This step loads the model (may take 30-60 seconds)...")
print("   Skip? (press Ctrl+C to skip, Enter to continue)")

try:
    input()
    from uncertainty.generation.generator import QwenGenerator

    generator = QwenGenerator(config.model)
    print(f"   [OK] Model loaded successfully")
    print(f"   [OK] GPU memory: {generator.get_gpu_memory_mb():.1f} MiB")

    # Quick generation test
    print("\n   Testing quick generation...")
    test_prompt = "Question: What is 2+2?\nAnswer:"
    answer = generator.generate_single(
        prompt=test_prompt,
        temperature=0.1,
        max_new_tokens=10
    )
    print(f"   [OK] Generated answer: {answer}")

except KeyboardInterrupt:
    print("\n   [SKIP] Skipped model loading test")
except Exception as e:
    print(f"   [ERROR] Error loading model: {e}")

print("\n" + "=" * 80)
print("Component testing complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Review the output above for any errors")
print("2. Run full generation: python generate_data.py --num_questions 10")
print("3. Check output in data/generated/")
