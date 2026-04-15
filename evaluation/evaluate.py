"""
samvad/evaluation/evaluate.py

Computes all 13 evaluation metrics on the generated outputs from each
fine-tuned model and saves a comparison results table.

Metrics computed:
    BLEU            — n-gram precision vs reference
    ROUGE-1         — unigram overlap
    ROUGE-2         — bigram overlap
    ROUGE-L         — longest common subsequence
    METEOR          — alignment-based metric (handles synonyms)
    GLEU            — sentence-level BLEU variant
    Repetition Rate — fraction of repeated n-grams in output
    Flesch Reading Ease — readability score (0-100, higher = easier)
    CoSIM           — cosine similarity of sentence embeddings
    BERT Score      — contextual token similarity (F1)
    Toxicity        — fraction of outputs classified as toxic
    Novelty         — fraction of output n-grams not in input
    Diversity       — distinct-2: unique bigrams / total bigrams

Output:
    results/metrics/scores.csv      — full per-model metric table
    results/metrics/scores.json     — same data as JSON

Usage (via main.py):
    python main.py --evaluate

Usage (directly):
    python evaluation/evaluate.py
"""

import json
import logging
import os
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn
from detoxify import Detoxify
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease

from config import config

logger = logging.getLogger(__name__)


RESULTS_DIR   = config.get("paths.results_dir")
GENERATED_DIR = os.path.join(RESULTS_DIR, "generated")
METRICS_DIR   = os.path.join(RESULTS_DIR, "metrics")

MODEL_NAMES = ["full_finetune", "lora", "qlora", "prefix_tuning"]

# Download required NLTK data
def download_nltk_data():
    for pkg in ["punkt", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

def compute_bleu(references: list, hypotheses: list) -> float:
    """Corpus-level BLEU with smoothing."""
    ref_tokens  = [[r.split()] for r in references]
    hyp_tokens  = [h.split() for h in hypotheses]
    smoothie    = SmoothingFunction().method4
    return corpus_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)


def compute_rouge(references: list, hypotheses: list) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rl),
    }


def compute_meteor(references: list, hypotheses: list) -> float:
    """Mean METEOR score."""
    scores = [
        meteor_score([ref.split()], hyp.split())
        for ref, hyp in zip(references, hypotheses)
    ]
    return np.mean(scores)


def compute_gleu(references: list, hypotheses: list) -> float:
    """Mean sentence-level GLEU score."""
    scores = [
        sentence_gleu([ref.split()], hyp.split())
        for ref, hyp in zip(references, hypotheses)
    ]
    return np.mean(scores)


def compute_repetition_rate(hypotheses: list, n: int = 4) -> float:
    """
    Fraction of repeated n-grams within each generated output.
    High repetition rate = model is looping or being repetitive.
    """
    rates = []
    for hyp in hypotheses:
        tokens = hyp.split()
        if len(tokens) < n:
            rates.append(0.0)
            continue
        ngrams      = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total       = len(ngrams)
        unique      = len(set(ngrams))
        repeated    = total - unique
        rates.append(repeated / total if total > 0 else 0.0)
    return np.mean(rates)


def compute_flesch(hypotheses: list) -> float:
    """Mean Flesch Reading Ease score (0–100, higher = more readable)."""
    scores = []
    for hyp in hypotheses:
        try:
            scores.append(flesch_reading_ease(hyp))
        except Exception:
            scores.append(0.0)
    return np.mean(scores)


def compute_cosim(
    references: list,
    hypotheses: list,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """
    Mean cosine similarity between reference and hypothesis sentence embeddings.
    Higher = semantically closer to the reference counterspeech.
    """
    logger.info("  Computing CoSIM embeddings …")
    embedder   = SentenceTransformer(model_name)
    ref_embs   = embedder.encode(references, batch_size=32, show_progress_bar=False)
    hyp_embs   = embedder.encode(hypotheses, batch_size=32, show_progress_bar=False)
    sims       = [
        cosine_similarity([r], [h])[0][0]
        for r, h in zip(ref_embs, hyp_embs)
    ]
    return float(np.mean(sims))


def compute_bert_score(references: list, hypotheses: list) -> float:
    """BERTScore F1 — contextual token-level similarity."""
    logger.info("  Computing BERTScore …")
    _, _, F1 = bert_score_fn(
        hypotheses,
        references,
        lang        = "en",
        rescale_with_baseline = True,
        verbose     = False,
    )
    return float(F1.mean())


def compute_toxicity(hypotheses: list) -> float:
    """
    Fraction of generated outputs classified as toxic.
    Lower is better — we want the model to reduce toxicity.
    Uses Detoxify's 'original' model.
    """
    logger.info("  Computing toxicity …")
    detector  = Detoxify("original")
    scores    = detector.predict(hypotheses)["toxicity"]
    # Toxicity score > 0.5 is classified as toxic
    toxic_frac = np.mean([s > 0.5 for s in scores])
    return float(toxic_frac)


def compute_novelty(hate_speeches: list, hypotheses: list, n: int = 2) -> float:
    """
    Fraction of output n-grams not present in the input hate speech.
    Higher = model is generating genuinely new content, not parroting input.
    """
    novelty_scores = []
    for src, hyp in zip(hate_speeches, hypotheses):
        src_ngrams = set(
            tuple(src.split()[i:i+n])
            for i in range(len(src.split()) - n + 1)
        )
        hyp_tokens = hyp.split()
        if len(hyp_tokens) < n:
            novelty_scores.append(1.0)
            continue
        hyp_ngrams = [
            tuple(hyp_tokens[i:i+n])
            for i in range(len(hyp_tokens) - n + 1)
        ]
        if not hyp_ngrams:
            novelty_scores.append(1.0)
            continue
        novel = sum(1 for g in hyp_ngrams if g not in src_ngrams)
        novelty_scores.append(novel / len(hyp_ngrams))
    return float(np.mean(novelty_scores))


def compute_diversity(hypotheses: list, n: int = 2) -> float:
    """
    Distinct-2: ratio of unique bigrams to total bigrams across all outputs.
    Higher = more diverse vocabulary across the generated set.
    """
    all_ngrams    = []
    unique_ngrams = set()
    for hyp in hypotheses:
        tokens = hyp.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)
    if not all_ngrams:
        return 0.0
    return len(unique_ngrams) / len(all_ngrams)


def evaluate_model(name: str, df: pd.DataFrame) -> dict:
    """Compute all 13 metrics for one model's generated outputs."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name}  ({len(df)} samples)")
    logger.info(f"{'='*60}")

    refs  = df["reference"].tolist()
    hyps  = df["generated"].tolist()
    srcs  = df["hate_speech"].tolist()

    # Replace empty generations with a placeholder to avoid division errors
    hyps = [h if h.strip() else "[empty]" for h in hyps]

    metrics = {}

    logger.info("  BLEU …")
    metrics["BLEU"] = round(compute_bleu(refs, hyps), 4)

    logger.info("  ROUGE …")
    rouge = compute_rouge(refs, hyps)
    metrics["ROUGE-1"] = round(rouge["rouge1"], 4)
    metrics["ROUGE-2"] = round(rouge["rouge2"], 4)
    metrics["ROUGE-L"] = round(rouge["rougeL"], 4)

    logger.info("  METEOR …")
    metrics["METEOR"] = round(compute_meteor(refs, hyps), 4)

    logger.info("  GLEU …")
    metrics["GLEU"] = round(compute_gleu(refs, hyps), 4)

    logger.info("  Repetition Rate …")
    metrics["Repetition Rate"] = round(compute_repetition_rate(hyps), 4)

    logger.info("  Flesch Reading Ease …")
    metrics["Flesch Reading Ease"] = round(compute_flesch(hyps), 2)

    metrics["CoSIM"]       = round(compute_cosim(refs, hyps), 4)
    metrics["BERT Score"]  = round(compute_bert_score(refs, hyps), 4)
    metrics["Toxicity"]    = round(compute_toxicity(hyps), 4)

    logger.info("  Novelty …")
    metrics["Novelty"]     = round(compute_novelty(srcs, hyps), 4)

    logger.info("  Diversity …")
    metrics["Diversity"]   = round(compute_diversity(hyps), 4)

    logger.info(f"  Done. BLEU={metrics['BLEU']} ROUGE-L={metrics['ROUGE-L']} BERTScore={metrics['BERT Score']}")
    return metrics


def evaluate(model_names: list = None) -> None:
    """
    Run evaluation for all (or selected) models.

    Args:
        model_names: Models to evaluate. Defaults to all four.
    """
    download_nltk_data()
    os.makedirs(METRICS_DIR, exist_ok=True)

    targets = model_names or MODEL_NAMES
    all_results = {}

    for name in targets:
        csv_path = os.path.join(GENERATED_DIR, f"{name}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"[{name}] No generated outputs at {csv_path} — skipping.")
            logger.warning(f"  Run: python main.py --generate first.")
            continue

        df = pd.read_csv(csv_path)
        all_results[name] = evaluate_model(name, df)

    if not all_results:
        logger.error("No results to save. Run --generate first.")
        return

    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"

    # Reorder columns to match project PDF table order
    col_order = [
        "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
        "METEOR", "GLEU", "Repetition Rate", "Flesch Reading Ease",
        "CoSIM", "BERT Score", "Toxicity", "Novelty", "Diversity",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Save
    csv_out  = os.path.join(METRICS_DIR, "scores.csv")
    json_out = os.path.join(METRICS_DIR, "scores.json")

    results_df.to_csv(csv_out)
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS TABLE")
    logger.info(f"{'='*60}")
    logger.info(f"\n{results_df.to_string()}")
    logger.info(f"\nSaved to:")
    logger.info(f"  {csv_out}")
    logger.info(f"  {json_out}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    evaluate()
