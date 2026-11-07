# Quick Start Guide - Semantic Entropy Data Generation

## What You Have Now

A complete data generation pipeline for reproducing the semantic entropy method from Farquhar et al. (Nature 2024) using Qwen3-1.7B.

## File Structure Created

```
Hallucination/
├── src/
│   ├── __init__.py              ✓ Package initialization
│   ├── config.py                ✓ Configuration management
│   ├── data_loader.py           ✓ TriviaQA dataset loading
│   ├── prompt_builder.py        ✓ Few-shot prompt construction
│   └── generator.py             ✓ Qwen3 model wrapper
├── data/
│   └── generated/               ✓ Output directory (created)
├── generate_data.py             ✓ Main generation script
├── test_pipeline.py             ✓ Component testing script
├── test_qwen.py                 ✓ Original model test
├── requirements.txt             ✓ Python dependencies
├── IMPLEMENTATION_PLAN.md       ✓ Detailed methodology
└── README.md                    ✓ Project documentation
```

## Step-by-Step Usage

### 1. Verify Environment

Check that you're in the correct virtual environment:

```bash
# Windows
hallucination\Scripts\activate

# Verify Python
python --version  # Should be 3.8+
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you already have these packages installed from your previous setup, this will just verify versions.

### 3. Test Components (Recommended)

Run the pipeline test to verify everything works:

```bash
python test_pipeline.py
```

This will:
- ✓ Test configuration loading
- ✓ Download and load TriviaQA dataset (first run only)
- ✓ Test prompt construction
- ✓ Optionally test model loading (you can skip this if you want)

**Expected output**: All tests should pass with ✓ marks.

### 4. Run Small Test Generation

Start with a small test (10 questions, ~5-10 minutes):

```bash
python generate_data.py --num_questions 10 --num_generations 5
```

This will:
1. Load TriviaQA dataset
2. Select 5 random few-shot examples
3. Load Qwen3-1.7B model (4-bit quantized)
4. For each of 10 questions:
   - Generate 1 low-temperature answer (temp=0.1)
   - Generate 5 high-temperature answers (temp=1.0)
5. Save results to `data/generated/trivia_qa_qwen3_YYYYMMDD_HHMMSS.json`

**Expected time**: ~1 minute per question = ~10 minutes total

### 5. Run Full Generation (Paper Scale)

Once the test works, run the full experiment (100 questions):

```bash
python generate_data.py --num_questions 100 --num_generations 10
```

**Expected time**: ~1.5-2 hours
**Output size**: ~5-10 MB JSON file

### 6. Scale Up (Optional)

For paper-scale experiments (400 questions):

```bash
python generate_data.py --num_questions 400 --num_generations 10
```

**Expected time**: ~6-8 hours

## Command-Line Options

```bash
python generate_data.py [OPTIONS]

Options:
  --dataset          Dataset to use (default: trivia_qa)
  --num_questions    Number of questions (default: 100)
  --num_few_shot     Few-shot examples (default: 5)
  --num_generations  High-temp samples (default: 10)
  --temperature      Sampling temperature (default: 1.0)
  --seed             Random seed (default: 42)
  --output_dir       Output directory (default: data/generated)
```

## What the Output Contains

Each generated file includes:

### Metadata Section
- Timestamp
- Complete configuration
- Few-shot prompt used
- Few-shot example indices (for reproducibility)

### Generations Section
For each question:
- Question text
- Reference answers (ground truth)
- Low-temperature generation (accuracy baseline)
- 10 high-temperature generations (for semantic entropy)

### Example Output Structure

```json
{
  "metadata": {
    "timestamp": "2026-10-24T...",
    "config": {
      "model": {"model_name": "Qwen3-1.7B", ...},
      "generation": {"temperature_high": 1.0, ...},
      "data": {"num_questions": 100, ...}
    },
    "few_shot_prompt": "Answer the following question...",
    "num_questions": 100
  },
  "generations": {
    "question_12345": {
      "question": "What is the capital of France?",
      "reference_answers": ["Paris"],
      "low_temp_generation": {
        "response": "Paris",
        "temperature": 0.1
      },
      "high_temp_generations": {
        "responses": [
          {"response": "Paris", "index": 0},
          {"response": "Paris is the capital", "index": 1},
          {"response": "The capital is Paris", "index": 2},
          ...
        ],
        "temperature": 1.0,
        "num_generations": 10
      }
    },
    ...
  }
}
```

## Next Steps After Generation

Once you have generated data, the next implementation steps are:

### Step 3: Semantic Clustering
- Implement bidirectional entailment checking
- Use DeBERTa NLI model to cluster responses
- Group semantically equivalent answers

### Step 4: Semantic Entropy Calculation
- Compute entropy over semantic clusters
- Compare with baseline methods (predictive entropy, etc.)

### Step 5: Evaluation
- Evaluate hallucination detection performance
- Compare with reference answers
- Analyze results

**Important**: Do not implement Steps 3-5 yet. Focus on generating quality data first.

## Troubleshooting

### "CUDA out of memory"
- Reduce `--num_questions`
- The model is already 4-bit quantized, so this shouldn't happen often
- Close other GPU applications

### "Dataset download failed"
- Ensure internet connection
- The dataset (~500 MB) downloads on first run
- Subsequent runs use cached data

### "Model not found"
- Verify model files are in: `C:\Users\TIRTH\OneDrive\Desktop\Hallucination\`
- Check that `config.json` exists in that directory

### Generation is slow
- Expected: ~6 seconds per question (1 low + 10 high temp generations)
- 4-bit quantization is already applied
- This is normal for a 1.7B model on RTX 2050

## Monitoring Progress

The script outputs:
- Current question number
- Question text
- Low-temperature answer
- Progress bar for all questions
- GPU memory usage

You can safely interrupt (Ctrl+C) and restart - each run is independent.

## Success Criteria

After running `generate_data.py`, you should have:

✓ A JSON file in `data/generated/`
✓ Metadata with configuration and few-shot prompt
✓ N questions with 1 low-temp + 10 high-temp answers each
✓ Reference answers for evaluation
✓ No errors in the generation log

## Questions?

Refer to:
- `README.md` - Project overview
- `IMPLEMENTATION_PLAN.md` - Detailed methodology
- Paper: https://arxiv.org/abs/2302.09664
- Official code: https://github.com/jlko/semantic_uncertainty

---

**Status**: Ready to run! 🚀

Start with: `python test_pipeline.py`
