# References

1. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. (2024). *Detecting hallucinations in large language
   models using semantic entropy.* Nature 630, 625-630.
   <https://www.nature.com/articles/s41586-024-07421-0>
2. Official semantic-uncertainty implementation: <https://github.com/jlko/semantic_uncertainty>
3. Kuhn, L., Gal, Y., Farquhar, S. (2023). *Semantic uncertainty: linguistic invariances for
   uncertainty estimation in natural language generation.* ICLR 2023. <https://arxiv.org/abs/2302.09664>
4. Joshi, M., Choi, E., Weld, D., Zettlemoyer, L. (2017). *TriviaQA: a large scale distantly
   supervised challenge dataset for reading comprehension.* ACL 2017.
   Dataset: <https://huggingface.co/datasets/mandarjoshi/trivia_qa>
5. Qwen Team (2025). *Qwen3 technical report.* Model: <https://huggingface.co/Qwen/Qwen3-1.7B>
6. NLI model: `cross-encoder/nli-deberta-v3-small`
   <https://huggingface.co/cross-encoder/nli-deberta-v3-small>

This repository is an experimental adaptation: it uses empirical cluster-frequency entropy, not the
probability-weighted estimator of reference 1.
