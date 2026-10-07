"""
Retrains the industry classifier on the FULL resume dataset with a held-out test split.

Outputs (originals are left untouched):
    industry_classifier_v2.pkl
    pca_transformer_v2.pkl
    training_report.txt
"""

import glob
import os

import joblib
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"D:\Computer Science\4th Year\Maam chatto\data"
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.npz")
REPORT_PATH = os.path.join(BASE_DIR, "training_report.txt")


def extract_text(pdf_path: str) -> str:
    try:
        reader = pypdf.PdfReader(pdf_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def load_dataset():
    """Returns (texts, labels, file_names) for every readable PDF in DATA_DIR."""
    texts, labels, names = [], [], []
    for category in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
            text = extract_text(path)
            if len(text.strip()) > 50:
                texts.append(text)
                labels.append(category)
                names.append(os.path.basename(path))
    return texts, np.array(labels), names


def main():
    texts, labels, names = load_dataset()
    print(f"Readable resumes: {len(texts)} across {len(set(labels))} industries")

    # Reuse embeddings if already computed for the same file list
    if os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH, allow_pickle=True)
        if list(cache["names"]) == names:
            embeddings = cache["embeddings"]
            print("Loaded cached embeddings.")
        else:
            cache = None
    else:
        cache = None

    if cache is None:
        print("Encoding resumes with all-MiniLM-L6-v2 (one-time cost)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True)
        np.savez(CACHE_PATH, embeddings=embeddings, names=np.array(names, dtype=object))

    # Held-out evaluation: PCA is fit on the training split only (no leakage)
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=42
    )

    pca = PCA(n_components=50, random_state=42)
    X_train = pca.fit_transform(embeddings[train_idx])
    X_test = pca.transform(embeddings[test_idx])

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, labels[train_idx])

    preds = clf.predict(X_test)
    acc = accuracy_score(labels[test_idx], preds)
    report = classification_report(labels[test_idx], preds, zero_division=0)
    summary = (
        f"Held-out test accuracy: {acc:.3f}  "
        f"(test size={len(test_idx)}, train size={len(train_idx)})\n\n{report}"
    )
    print(summary)

    # Final deployment model: refit PCA and classifier on all data
    final_pca = PCA(n_components=50, random_state=42)
    X_all = final_pca.fit_transform(embeddings)
    final_clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    final_clf.fit(X_all, labels)

    joblib.dump(final_clf, os.path.join(BASE_DIR, "industry_classifier_v2.pkl"))
    joblib.dump(final_pca, os.path.join(BASE_DIR, "pca_transformer_v2.pkl"))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved v2 models and {REPORT_PATH}")


if __name__ == "__main__":
    main()
