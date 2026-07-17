"""
baseline.py
Part A: TF-IDF Retrieval + FLAN-T5-base extraction baseline on SQuAD 2.0.

Pipeline:
  question -> TF-IDF retriever (over unique contexts) -> top-1 context
  (question, context) -> prompt -> google/flan-t5-base -> answer string
  answer vs gold -> Exact Match (EM) / F1 (SQuAD-style)
"""

import re
import string
import collections
import random
import json

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

random.seed(42)
np.random.seed(42)

SAMPLE_SIZE = 150


def load_sample():
    """Load a fixed, reproducible sample of SQuAD 2.0 validation, answerable
    questions only (unanswerable ones have no gold span, so EM/F1 against them
    is not meaningful for this experiment)."""
    ds = load_dataset("squad_v2", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
    ds = ds.shuffle(seed=42).select(range(SAMPLE_SIZE))
    return ds


class TfidfRetriever:
    """Minimal TF-IDF retriever over the pool of unique context paragraphs."""

    def __init__(self, contexts):
        self.contexts = contexts
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(contexts)

    def retrieve(self, query, top_k=1):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.contexts[i] for i in top_idx], sims[top_idx]


# ---------------------------------------------------------------------------
# SQuAD-style normalization / EM / F1 (standard implementation)
# ---------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_metric_over_ground_truths(metric_fn, pred, golds):
    return max(metric_fn(pred, g) for g in golds)


def build_prompt(question, context):
    return (
        "Answer the question using only the context below. "
        "If the answer cannot be found, respond with 'unanswerable'.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def run_baseline():
    print("Loading SQuAD 2.0 validation sample...")
    ds = load_sample()

    print("Building TF-IDF retriever over unique contexts...")
    unique_contexts = list(dict.fromkeys(ds["context"]))
    retriever = TfidfRetriever(unique_contexts)

    print("Loading google/flan-t5-base ...")
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=32)

    em_scores, f1_scores = [], []
    logs = []

    for i, example in enumerate(ds):
        question = example["question"]
        gold_context = example["context"]
        gold_answers = example["answers"]["text"]

        retrieved_contexts, _sims = retriever.retrieve(question, top_k=1)
        retrieved_context = retrieved_contexts[0]
        retrieval_hit = int(retrieved_context.strip() == gold_context.strip())

        prompt = build_prompt(question, retrieved_context)
        output = qa_pipe(prompt)[0]["generated_text"].strip()

        em = max_metric_over_ground_truths(compute_em, output, gold_answers)
        f1 = max_metric_over_ground_truths(compute_f1, output, gold_answers)

        em_scores.append(em)
        f1_scores.append(f1)

        logs.append({
            "id": example["id"],
            "question": question,
            "gold_answers": gold_answers,
            "retrieval_hit": retrieval_hit,
            "prediction": output,
            "em": em,
            "f1": round(f1, 3),
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(ds)} examples...")

    avg_em = 100 * np.mean(em_scores)
    avg_f1 = 100 * np.mean(f1_scores)
    retrieval_acc = 100 * np.mean([l["retrieval_hit"] for l in logs])

    print("\n===== BASELINE RESULTS (Part A) =====")
    print(f"Sample size:        {len(ds)}")
    print(f"Retrieval accuracy: {retrieval_acc:.2f}%")
    print(f"Exact Match (EM):   {avg_em:.2f}%")
    print(f"F1 Score:           {avg_f1:.2f}%")

    with open("baseline_results.json", "w") as f:
        json.dump({
            "summary": {"em": avg_em, "f1": avg_f1, "retrieval_acc": retrieval_acc},
            "logs": logs,
        }, f, indent=2)
    print("\nSaved detailed logs to baseline_results.json")


if __name__ == "__main__":
    run_baseline()