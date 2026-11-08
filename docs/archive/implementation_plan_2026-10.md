> **ARCHIVED — historical document.** Original implementation plan (October 2026). It is kept for the record only and
> is **superseded** by the current [research summary](../research_summary.md),
> [experiment index](../experiments/experiment_index.md) and
> [methodology](../methodology/README.md). File paths, commands, statuses and
> "pending" items below are out of date.

# Semantic Entropy Implementation Plan

## Paper: "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
**Authors**: Farquhar et al., Nature 2024

## Current Status: Step 2 - Data Generation Stage

---

## Key Findings from Paper Analysis

### 1. Datasets Used
- **TriviaQA** (primary choice for initial experiments)
- SQuAD v2
- BioASQ
- Natural Questions (NQ)
- SVAMP (math word problems)

### 2. Generation Parameters (from code analysis)
- **Temperature for sampling**: `1.0` (default, configurable)
- **Temperature for accuracy baseline**: `0.1` (low-temperature, near-greedy)
- **Number of sampled generations**: `10` (default via `--num_generations=10`)
- **Top-p**: Not explicitly set (uses model defaults)
- **Top-k**: Not explicitly set (uses model defaults)
- **Max new tokens**: `50` (default, for short-phrase answers)

### 3. Prompt Structure (Few-Shot)
The paper uses a **5-shot** prompting setup by default:

```
Answer the following question as briefly as possible.

Question: [example question 1]
Answer: [example answer 1]

Question: [example question 2]
Answer: [example answer 2]

[... 3 more examples ...]

Question: [actual question]
Answer:
```

### 4. Generation Strategy
For each question, the code generates:
1. **One low-temperature generation** (temp=0.1): Used for accuracy evaluation
2. **N high-temperature generations** (temp=1.0, N=10): Used for semantic entropy calculation

### 5. Data Storage Format
The paper stores results as nested dictionaries with this structure:

```python
generations = {
    'question_id_1': {
        'question': 'What is ...',
        'context': None,  # or context string for SQuAD
        'reference': {
            'id': 'question_id_1',
            'answers': {'text': ['answer1', 'answer2'], 'answer_start': [0, 0]}
        },
        'most_likely_answer': {
            'response': 'low-temp answer',
            'token_log_likelihoods': [log_probs],
            'embedding': tensor,
            'accuracy': 1.0 or 0.0
        },
        'responses': [
            ('high-temp answer 1', [log_probs], embedding, accuracy),
            ('high-temp answer 2', [log_probs], embedding, accuracy),
            # ... 10 total
        ]
    },
    'question_id_2': { ... },
    # ...
}
```

Saved as pickle files via `wandb` (Weights & Biases) logging system.

---

## Adaptation for Qwen3-1.7B

### Differences to Account For:
1. **Model Architecture**: Qwen3 vs Llama-2/Falcon (paper models)
2. **Quantization**: 4-bit quantization (we're using) vs full precision
3. **Thinking Mode**: Qwen3 has thinking mode - we should **disable it** (`enable_thinking=False`)
4. **Generation Config**: Need to explicitly set temperature, top-p, top-k

### Recommended Settings:
```python
# For non-thinking mode (per Qwen3 documentation):
temperature = 1.0  # For sampling (matches paper)
top_p = 0.8       # Recommended for Qwen3 non-thinking
top_k = 20        # Recommended for Qwen3
temperature_low = 0.1  # For accuracy baseline

# For generation:
max_new_tokens = 64  # Slightly higher than paper's 50
do_sample = True     # Enable sampling (not greedy)
```

---

## Implementation Steps

### ✅ Step 1: Environment Setup (COMPLETED)
- Virtual environment `hallucination` exists
- Qwen3-1.7B model downloaded locally
- 4-bit quantization working
- Test script confirms model loading and inference

### 🔄 Step 2: Data Generation Pipeline (CURRENT)

#### 2.1 Dataset Selection and Loading
**Choice**: Start with **TriviaQA**
- Widely used QA benchmark
- Short factual answers (ideal for semantic clustering)
- Available via HuggingFace: `TimoImhof/TriviaQA-in-SQuAD-format`

**Implementation**:
- Load TriviaQA dataset
- Start with **100 questions** for initial testing
- Use validation split for evaluation

#### 2.2 Few-Shot Prompt Construction
- Select 5 random examples from training set
- Format as shown above
- Store prompt indices for reproducibility

#### 2.3 Generation Loop
For each question:
1. Generate 1 low-temperature answer (temp=0.1)
2. Generate 10 high-temperature answers (temp=1.0)
3. Store all responses with metadata

#### 2.4 Data Storage
- Use JSON format (more portable than pickle)
- Store generation config alongside data
- Include timestamp and model info

### ⏳ Step 3: Semantic Clustering (NOT YET)
- Will use NLI model (DeBERTa) for bidirectional entailment
- Cluster semantically equivalent answers
- **DO NOT IMPLEMENT YET** - wait until Step 2 is complete

### ⏳ Step 4: Semantic Entropy Calculation (NOT YET)
- Compute entropy over semantic clusters
- Compare with baseline methods
- **DO NOT IMPLEMENT YET**

### ⏳ Step 5: Evaluation (NOT YET)
- Evaluate hallucination detection performance
- Compare predictions with ground truth
- **DO NOT IMPLEMENT YET**

---

## File Structure (Proposed)

```
Hallucination/
├── test_qwen.py                    # Existing test script
├── config.json                     # Model config
├── generation_config.json          # Model generation config
├── model files...                  # Existing model weights
├── semantic_uncertainty_repo/      # Cloned reference repo
├── data/
│   ├── triviaqa_raw/              # Downloaded dataset
│   ├── few_shot_examples.json     # Selected few-shot examples
│   └── generated/                  # Generated responses
│       └── triviaqa_qwen3_run1.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading utilities
│   ├── prompt_builder.py          # Few-shot prompt construction
│   ├── generator.py               # Multi-response generation
│   └── config.py                  # Configuration management
├── notebooks/
│   └── explore_data.ipynb         # Data exploration
├── IMPLEMENTATION_PLAN.md         # This file
└── README.md                       # Project documentation
```

---

## Next Actions

1. **Create data generation pipeline**:
   - `src/config.py` - Configuration management
   - `src/data_loader.py` - Load TriviaQA dataset
   - `src/prompt_builder.py` - Build few-shot prompts
   - `src/generator.py` - Generate multiple responses per question

2. **Run initial experiment**:
   - Generate responses for 100 TriviaQA questions
   - Validate data format and storage
   - Check generation quality manually

3. **Scale up**:
   - Increase to 400 questions (paper default)
   - Monitor GPU memory usage
   - Track generation time

4. **Document findings**:
   - Generation quality observations
   - Any Qwen3-specific quirks
   - Prompt effectiveness

---

## References

- Paper: https://arxiv.org/abs/2302.09664
- Official Code: https://github.com/jlko/semantic_uncertainty
- Qwen3 Documentation: https://qwen.readthedocs.io/
- TriviaQA Dataset: https://huggingface.co/datasets/TimoImhof/TriviaQA-in-SQuAD-format

---

## Notes

- The paper's semantic entropy method is **unsupervised** - no labeled hallucination data needed
- Focus on **reproducing the method faithfully** before attempting modifications
- Keep generated data separate from clustering/entropy stages
- Track all hyperparameters for reproducibility
