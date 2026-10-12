QA Robustness to Adversarial Distractors

**By Harsh Yadav**

A two-part study on SQuAD 2.0:

- **Part A (`baseline.py`)** — a simple TF-IDF retriever + `google/flan-t5-base`
  extractive/generative QA pipeline, scored with Exact Match (EM) and F1.
- **Part B (`experiment.py`)** — tests the hypothesis that injecting a
  lexically-similar, factually contradictory "adversarial distractor"
  sentence into the retrieved context breaks the model's reasoning, even
  though retrieval itself is untouched.

## Prerequisites

Python 3.9+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: the first run will download `google/flan-t5-base` (~1GB) and the
> `squad_v2` dataset from Hugging Face Hub — an internet connection is
> required the first time.

## How to run

**Part A — Baseline:**

```bash
python baseline.py
```

Prints retrieval accuracy, EM, and F1 to stdout, and writes per-example logs
to `baseline_results.json`.

**Part B — Adversarial Distractor Experiment:**

```bash
python experiment.py
```

Runs the same 150-example sample (same random seed as `baseline.py`) through
both a clean-context condition and a distractor-injected condition, prints
the clean vs. adversarial EM/F1 and the "confidently fooled" rate, and
writes:
- `experiment_results.json` — full per-example logs (clean + adversarial)
- `fooled_examples.json` — only the cases where the model was correct on
  the clean context but flipped to the injected fake answer

## Results summary

| Condition | EM | F1 |
|---|---|---|
| Baseline (clean context) | 66.67% | 77.67% |
| + Adversarial distractor | 60.67% | 71.57% |
| Δ (drop) | -6.00 pts | -6.10 pts |

Full analysis, hypothesis, and next steps are in [`writeup.pdf`](./writeup.pdf).

## Design notes

- Both scripts fix `random.seed(42)` / dataset `shuffle(seed=42)` so the same
  150 questions are used in both parts, isolating the distractor as the only
  variable.
- Only answerable SQuAD 2.0 questions are sampled, since EM/F1 need a gold
  span to compare against.
- The distractor generator clones the gold-answer sentence verbatim and only
  swaps the answer span, maximizing lexical/syntactic overlap with the true
  answer sentence — this is the core adversarial mechanism under test (see
  `writeup.pdf`).
