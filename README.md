# Semantic Entropy for Hallucination Detection

Implementation of "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Farquhar et al., Nature 2024) using Qwen3-1.7B.

## Project Overview

This project reproduces the semantic entropy method for detecting hallucinations in LLM outputs using a smaller, quantized model (Qwen3-1.7B with 4-bit quantization).

**Current Stage**: Step 2 - Data Generation

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

**Step 2 (Current): Data Generation**

For each question:
- Generate 1 low-temperature answer (temp=0.1) for accuracy baseline
- Generate 10 high-temperature answers (temp=1.0) for entropy estimation
- Store all responses with metadata

**Next Steps:**
- Step 3: Semantic clustering using DeBERTa NLI
- Step 4: Semantic entropy calculation
- Step 5: Evaluation and hallucination detection

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
