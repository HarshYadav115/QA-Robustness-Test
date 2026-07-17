# CortiqoLabs NLP/GenAI Take-Home — QA Robustness to Adversarial Distractors

A two-part study on SQuAD 2.0:

- **Part A (`baseline.py`)** — a simple TF-IDF retriever + `google/flan-t5-base`
  extractive/generative QA pipeline, scored with Exact Match (EM) and F1.
- **Part B (`experiment.py`)** — tests the hypothesis that injecting a
  lexically-similar, factually contradictory "adversarial distractor"
  sentence into the retrieved context breaks the model's reasoning, even
  though retrieval itself is untouched.

## Repository structure