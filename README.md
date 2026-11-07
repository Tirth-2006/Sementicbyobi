# Semantic Entropy for Hallucination Detection

Implementation of "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Farquhar et al., Nature 2024) using Qwen3-1.7B.

## Project Overview

This project reproduces the semantic entropy method for detecting hallucinations in LLM outputs using a smaller, quantized model (Qwen3-1.7B with 4-bit quantization).

**Current Stage**: Evaluation complete for the saved 100-question artifact; ongoing reproducibility and sensitivity analysis

## Setup

### Prerequisites

- Python 3.8+ with virtual environment `hallucination`
- CUDA-capable GPU (tested on NVIDIA RTX 2050)
- 16 GB system RAM
- Qwen3-1.7B model files (downloaded locally)

### Installation

1. Activate virtual environment:
```bash
# Assuming your virtual environment is already set up
# If not, create it first:
# python -m venv hallucination
# hallucination\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install torch transformers accelerate bitsandbytes datasets tqdm
```

## Usage

### Generate Semantic Entropy Dataset

Run the data generation pipeline:

```bash
python generate_data.py --num_questions 100 --num_generations 10
```

**Arguments:**
- `--dataset`: Dataset to use (default: `trivia_qa`)
- `--num_questions`: Number of questions to process (default: 100)
- `--num_few_shot`: Number of few-shot examples (default: 5)
- `--num_generations`: Number of high-temperature samples (default: 10)
- `--temperature`: Sampling temperature (default: 1.0)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory (default: `data/generated`)

### Output Format

Generated data is saved as JSON:

```json
{
  "metadata": {
    "timestamp": "2026-10-24T...",
    "config": {...},
    "few_shot_prompt": "...",
    "num_questions": 100
  },
  "generations": {
    "question_id_1": {
      "question": "What is...",
      "reference_answers": ["answer1", "answer2"],
      "low_temp_generation": {
        "response": "...",
        "temperature": 0.1
      },
      "high_temp_generations": {
        "responses": [
          {"response": "...", "index": 0},
          ...
        ],
        "temperature": 1.0,
        "num_generations": 10
      }
    }
  }
}
```

## Project Structure

```
Hallucination/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Dataset loading
│   ├── prompt_builder.py      # Few-shot prompt construction
│   └── generator.py           # Model wrapper and generation
├── data/
│   └── generated/             # Generated responses
├── generate_data.py           # Main generation script
├── test_qwen.py              # Model test script
├── IMPLEMENTATION_PLAN.md    # Detailed implementation plan
└── README.md                 # This file
```

## Methodology

### Paper: Farquhar et al. (Nature 2024)

The semantic entropy method:

1. **Sample multiple generations** per question (temperature = 1.0)
2. **Cluster by semantic equivalence** using NLI model
3. **Compute entropy over clusters** (not individual sequences)
4. High entropy → likely hallucination

### Our Implementation

**Data Generation**

For each question:
- Generate 1 low-temperature answer (temp=0.1) for accuracy baseline
- Generate 10 high-temperature answers (temp=1.0) for entropy estimation
- Store all responses with metadata

### Saved Evaluation Results

The saved 100-question evaluation artifact contains 27 correct and 73 incorrect
low-temperature answers, for an initial accuracy of **0.270000**. Higher entropy
answers are rejected first, with ties ordered by question ID. Using trapezoidal
integration over retained coverage:

| Estimator | AUROC | AURAC |
|---|---:|---:|
| Token entropy | 0.851852 | 0.537749 |
| Empirical semantic entropy | 0.841451 | 0.462878 |

For empirical semantic entropy, the direct random-rejection baseline is
**0.269635** (20,000 simulations; 95% interval **[0.187234, 0.360535]**).
The semantic-entropy AURAC exceeds this baseline by **0.193243**.

The detailed AURAC result is saved in
`data/generated/trivia_qa_qwen3_20261024_162539_aurac_results.json`, with the
corresponding rejection curve in
`data/generated/trivia_qa_qwen3_20261024_162539_aurac_curve.csv`.

### Hidden-State Dispersion Experiments

The frozen 100-question artifact also includes answer-level Qwen3 hidden-state
experiments. The original extractor uses predictor states at `p-1`; a separate
post-token extractor uses states at `p`. Both evaluate hidden-state index 28
with mean pooling and preserve the original NPZ artifacts.

The standalone scripts are:

- `evaluate_post_token_hidden_state_dispersion.py`
- `evaluate_post_token_dispersion_predictive.py`
- `evaluate_predictor_vs_post_token_dispersion.py`

The predictor-vs-post-token comparison uses the same question-level nested
cross-validation protocol as the hidden-state evaluator. In the 100-question
experiment, predictor-state dispersion gave AUROC **0.839168** and AURAC
**0.515165** when added to semantic plus token entropy, versus AUROC
**0.832065** and AURAC **0.509229** for post-token dispersion. The paired
20,000-resample confidence intervals included zero for the primary AUROC and
AURAC differences, so these are numerical, not statistically conclusive,
improvements.

## Configuration

Default generation parameters (following paper):

```python
temperature_low = 0.1      # Accuracy baseline
temperature_high = 1.0     # Sampling for entropy
top_p = 0.8               # Nucleus sampling
top_k = 20                # Top-k sampling
num_generations = 10       # Samples per question
max_new_tokens = 64       # Answer length limit
enable_thinking = False    # Disable Qwen3 thinking mode
```

## References

- **Paper**: [Detecting Hallucinations in Large Language Models Using Semantic Entropy](https://arxiv.org/abs/2302.09664)
- **Official Code**: [github.com/jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty)
- **Model**: Qwen3-1.7B ([Hugging Face](https://huggingface.co/Qwen/Qwen3-1.7B))
- **Dataset**: TriviaQA ([Hugging Face](https://huggingface.co/datasets/trivia_qa))

## License

This implementation is for research and educational purposes.

## Acknowledgments

- Farquhar et al. for the semantic entropy method
- Qwen team for the Qwen3 model
- TriviaQA dataset authors
