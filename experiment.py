"""
experiment.py
Part B: Adversarial Distractor Injection Experiment.

Hypothesis: Injecting a highly context-relevant but factually contradictory
"adversarial distractor" sentence into the retrieved context will cause
FLAN-T5-base to abandon the correct answer, revealing that the model relies
on superficial lexical/syntactic overlap rather than robust logical
extraction.

Method: for each example the baseline got right, we clone the gold sentence,
swap the answer span for a plausible-but-wrong "fake answer" of the same
surface type (number, date, proper noun), and append it with a contrastive
cue ("However, other records dispute this..."). Retrieval is held constant
so any metric drop is attributable to the injected sentence, not to a
retrieval failure.
"""

import re
import json
import random

import numpy as np
from transformers import pipeline

from baseline import (
    load_sample,
    TfidfRetriever,
    build_prompt,
    normalize_answer,
    compute_em,
    compute_f1,
    max_metric_over_ground_truths,
)

random.seed(42)
np.random.seed(42)

DECOY_NAMES = [
    "Marcus Whitfield", "the Delacroix Corporation", "Lindholm Industries",
    "Dr. Elena Sato", "the Bramwell Institute", "Novara Group", "Aldrich & Sons",
]
DECOY_PLACES = [
    "Reykjavik", "Porto Alegre", "Novosibirsk", "Christchurch", "Tallinn", "Bratislava",
]


def generate_fake_answer(answer_text):
    """Produce a same-surface-type but wrong replacement for the gold answer."""
    text = answer_text.strip()

    if re.fullmatch(r"\d{3,4}", text):
        year = int(text)
        fake_year = year + random.choice([-77, 42, 88, -33, 61, 15])
        return str(fake_year)

    if re.search(r"\d", text):
        nums = re.findall(r"\d+", text)
        new_text = text
        for n in nums:
            shift = random.choice([5, 10, -7, 23, -15])
            new_text = new_text.replace(n, str(max(int(n) + shift, 0)), 1)
        return new_text

    if text[:1].isupper():
        return random.choice(DECOY_NAMES + DECOY_PLACES)

    return "a completely unrelated entity"


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def build_distractor_sentence(context, answer):
    """Find the sentence containing the gold answer, clone it with the
    answer swapped for a fake one, and wrap it in a contrastive frame so it
    reads as contradicting evidence."""
    fake_answer = generate_fake_answer(answer)
    host_sentence = None
    for sent in split_sentences(context):
        if answer.lower() in sent.lower():
            host_sentence = sent
            break

    if host_sentence:
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        modified = pattern.sub(fake_answer, host_sentence, count=1)
    else:
        modified = f"the correct figure was in fact {fake_answer}."

    distractor_sentence = (
        f"However, other historical records dispute this and instead state that {modified}"
    )
    return distractor_sentence, fake_answer


def inject_distractor(context, answer):
    distractor_sentence, fake_answer = build_distractor_sentence(context, answer)
    poisoned_context = context.rstrip() + " " + distractor_sentence
    return poisoned_context, fake_answer, distractor_sentence


def fell_for_distractor(prediction, fake_answer, gold_em):
    """A 'confident fooling' event: the model got it right on the clean
    context, but its adversarial-context answer matches the injected fake
    answer instead of the gold answer."""
    if gold_em != 1:
        return False
    f1_vs_fake = compute_f1(prediction, fake_answer)
    substring_hit = normalize_answer(fake_answer) in normalize_answer(prediction)
    return bool(substring_hit or f1_vs_fake >= 0.5)


def run_experiment():
    print("Loading SQuAD 2.0 validation sample (same seed as baseline)...")
    ds = load_sample()

    print("Building TF-IDF retriever over unique contexts...")
    unique_contexts = list(dict.fromkeys(ds["context"]))
    retriever = TfidfRetriever(unique_contexts)

    print("Loading google/flan-t5-base ...")
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=32)

    clean_em_scores, clean_f1_scores = [], []
    adv_em_scores, adv_f1_scores = [], []
    fooled_count = 0
    logs = []

    for i, example in enumerate(ds):
        question = example["question"]
        gold_answers = example["answers"]["text"]
        primary_answer = gold_answers[0]

        retrieved_contexts, _sims = retriever.retrieve(question, top_k=1)
        clean_context = retrieved_contexts[0]

        # ---- Clean run ----
        clean_prompt = build_prompt(question, clean_context)
        clean_output = qa_pipe(clean_prompt)[0]["generated_text"].strip()
        clean_em = max_metric_over_ground_truths(compute_em, clean_output, gold_answers)
        clean_f1 = max_metric_over_ground_truths(compute_f1, clean_output, gold_answers)

        # ---- Adversarial run ----
        poisoned_context, fake_answer, distractor_sentence = inject_distractor(
            clean_context, primary_answer
        )
        adv_prompt = build_prompt(question, poisoned_context)
        adv_output = qa_pipe(adv_prompt)[0]["generated_text"].strip()
        adv_em = max_metric_over_ground_truths(compute_em, adv_output, gold_answers)
        adv_f1 = max_metric_over_ground_truths(compute_f1, adv_output, gold_answers)

        clean_em_scores.append(clean_em)
        clean_f1_scores.append(clean_f1)
        adv_em_scores.append(adv_em)
        adv_f1_scores.append(adv_f1)

        fooled = fell_for_distractor(adv_output, fake_answer, clean_em)
        if fooled:
            fooled_count += 1

        logs.append({
            "id": example["id"],
            "question": question,
            "gold_answers": gold_answers,
            "distractor_sentence": distractor_sentence,
            "fake_answer": fake_answer,
            "clean_prediction": clean_output,
            "clean_em": clean_em,
            "clean_f1": round(clean_f1, 3),
            "adversarial_prediction": adv_output,
            "adversarial_em": adv_em,
            "adversarial_f1": round(adv_f1, 3),
            "fell_for_distractor": fooled,
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(ds)} examples...")

    clean_em_avg = 100 * np.mean(clean_em_scores)
    clean_f1_avg = 100 * np.mean(clean_f1_scores)
    adv_em_avg = 100 * np.mean(adv_em_scores)
    adv_f1_avg = 100 * np.mean(adv_f1_scores)
    fooled_rate = 100 * fooled_count / len(ds)

    print("\n===== EXPERIMENT RESULTS (Part B: Adversarial Distractor) =====")
    print(f"Sample size:                                  {len(ds)}")
    print(f"Clean EM / F1:                                 {clean_em_avg:.2f}% / {clean_f1_avg:.2f}%")
    print(f"Adversarial EM / F1:                           {adv_em_avg:.2f}% / {adv_f1_avg:.2f}%")
    print(f"F1 drop:                                       {clean_f1_avg - adv_f1_avg:.2f} points")
    print(f"EM drop:                                        {clean_em_avg - adv_em_avg:.2f} points")
    print(f"Confidently fooled (of clean-correct cases):    {fooled_rate:.2f}%")

    fooled_examples = [l for l in logs if l["fell_for_distractor"]]
    print(f"\nLogging {len(fooled_examples)} confident-failure cases to fooled_examples.json")

    with open("experiment_results.json", "w") as f:
        json.dump({
            "summary": {
                "clean_em": clean_em_avg,
                "clean_f1": clean_f1_avg,
                "adversarial_em": adv_em_avg,
                "adversarial_f1": adv_f1_avg,
                "f1_drop": clean_f1_avg - adv_f1_avg,
                "em_drop": clean_em_avg - adv_em_avg,
                "fooled_rate_pct": fooled_rate,
            },
            "logs": logs,
        }, f, indent=2)

    with open("fooled_examples.json", "w") as f:
        json.dump(fooled_examples, f, indent=2)

    print("\nSaved full logs to experiment_results.json")


if __name__ == "__main__":
    run_experiment()