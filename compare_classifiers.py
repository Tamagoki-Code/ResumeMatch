"""
Compares industry classifier candidates with 5-fold stratified cross-validation,
then saves the best one as industry_model_v3.joblib.

Each fold fits its own TF-IDF vectorizer / PCA / scaler, so no test data leaks into training.
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import scoring
from train_industry_classifier import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "classifier_comparison.json")
TEXTS_PATH = os.path.join(BASE_DIR, "dataset_texts.json")


def get_data():
    if os.path.exists(TEXTS_PATH):
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            texts, labels = json.load(f)
        labels = np.array(labels)
    else:
        texts, labels, _ = load_dataset()
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump([texts, labels.tolist()], f)
    cache = np.load(CACHE_PATH, allow_pickle=True)
    return texts, labels, cache["embeddings"]


def cross_validate(kind, texts, labels, embs):
    texts = np.array(texts, dtype=object)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_pred, confidence = [], [], []
    for train_idx, test_idx in skf.split(embs, labels):
        bundle = scoring.fit_bundle(kind, list(texts[train_idx]), embs[train_idx], labels[train_idx])
        probs = scoring.predict_proba(bundle, list(texts[test_idx]), embs[test_idx])
        y_pred.extend(bundle["classes"][probs.argmax(axis=1)])
        confidence.extend(probs.max(axis=1))
        y_true.extend(labels[test_idx])
    y_true, y_pred, confidence = np.array(y_true), np.array(y_pred), np.array(confidence)

    confident = confidence >= scoring.UNCLEAR_BELOW
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confident_share": float(confident.mean()),
        "accuracy_when_confident": float(accuracy_score(y_true[confident], y_pred[confident])) if confident.any() else 0.0,
    }


def main():
    texts, labels, embs = get_data()
    print(f"Dataset: {len(texts)} resumes, {len(set(labels))} industries")

    candidates = ["tfidf", "sbert_lr", "tfidf_sbert", "sbert_pca_rf"]
    results = {}
    for kind in candidates:
        print(f"Running 5-fold CV: {kind} ...", flush=True)
        results[kind] = cross_validate(kind, texts, labels, embs)
        r = results[kind]
        print(f"  accuracy={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}  "
              f"confident={r['confident_share']:.2f}  acc_when_confident={r['accuracy_when_confident']:.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest by macro F1: {best}")

    bundle = scoring.fit_bundle(best, texts, embs, labels)
    out = os.path.join(BASE_DIR, "industry_model_v3.joblib")
    joblib.dump(bundle, out)
    print(f"Saved final model ({best}) trained on all data -> {out}")


if __name__ == "__main__":
    main()
