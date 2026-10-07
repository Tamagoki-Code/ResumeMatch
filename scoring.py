"""
Resume scoring logic for ResumeMatch AI.

Kept free of Streamlit so it can be unit-tested and reused by the training scripts.
The UI lives in app.py.
"""

import io
import os
import re

import joblib
import numpy as np
import pypdf
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SBERT_NAME = "all-MiniLM-L6-v2"
INDUSTRY_MODEL_PATH = os.path.join(BASE_DIR, "industry_model_v3.joblib")

# Calibration: raw hybrid similarity 0.20 -> 0%, 0.70 -> 100%.
CAL_LOW, CAL_HIGH = 0.20, 0.70
TIER_HIGH, TIER_MID = 75.0, 45.0
STRENGTH_MIN_SIM = 0.30
GAP_MAX_SIM = 0.40
MAX_GAPS = 5

# Below this top probability the industry prediction is reported as unclear.
UNCLEAR_BELOW = 0.35

KINDS = ("tfidf", "sbert_lr", "tfidf_sbert", "sbert_pca_rf")


class ExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


# =====================================================================
# PDF + text
# =====================================================================
def extract_text(file_bytes: bytes) -> str:
    """Returns the text of a PDF, or raises ExtractionError if it can't be read."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ExtractionError(f"Unreadable PDF: {exc}") from exc
    text = text.replace("\n", " ").strip()
    if len(text) <= 50:
        raise ExtractionError("PDF contains too little extractable text (scanned image?)")
    return text


def chunk_into_sentences(text: str) -> list:
    """Splits text into sentence chunks, dropping very short fragments."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?•]) +", text.replace("•", "."))]
    return [s for s in sentences if len(s.split()) > 3]


# =====================================================================
# Industry classifier: candidate feature sets and models
# =====================================================================
def _new_vectorizer():
    return TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
        stop_words="english", max_features=50000,
    )


def fit_bundle(kind: str, texts, embs, labels) -> dict:
    """Fits a classifier of the given kind. `embs` is a (n, 384) array of Sentence-BERT embeddings."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kind: {kind}")
    bundle = {"kind": kind, "vectorizer": None, "scaler": None, "pca": None}
    # The vectorizer is always fitted so every model can explain its prediction with terms.
    bundle["vectorizer"] = _new_vectorizer().fit(texts)
    if kind in ("sbert_lr", "tfidf_sbert"):
        bundle["scaler"] = StandardScaler().fit(embs)
    if kind == "sbert_pca_rf":
        bundle["pca"] = PCA(n_components=50, random_state=42).fit(embs)
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    elif kind == "tfidf":
        clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
    else:
        clf = LogisticRegression(C=2, max_iter=3000, class_weight="balanced")
    X = featurize(bundle, texts, embs)
    clf.fit(X, labels)
    bundle["clf"] = clf
    bundle["classes"] = clf.classes_
    return bundle


def featurize(bundle: dict, texts, embs):
    if bundle["kind"] == "sbert_pca_rf":
        return bundle["pca"].transform(embs)
    parts = []
    if bundle["kind"] in ("tfidf", "tfidf_sbert"):
        parts.append(bundle["vectorizer"].transform(texts))
    if bundle["scaler"] is not None:
        parts.append(csr_matrix(bundle["scaler"].transform(embs)))
    return hstack(parts).tocsr()


def predict_proba(bundle: dict, texts, embs) -> np.ndarray:
    return bundle["clf"].predict_proba(featurize(bundle, texts, embs))


def explain_industry(bundle: dict, text: str, label: str, n: int = 6):
    """
    Returns the resume terms that pushed the model toward `label`.
    Only available for linear TF-IDF models; returns [] otherwise.
    """
    if bundle["kind"] not in ("tfidf", "tfidf_sbert"):
        return []
    vec = bundle["vectorizer"]
    x = vec.transform([text]).tocoo()
    coef = bundle["clf"].coef_[list(bundle["classes"]).index(label)]
    contributions = [(vec.get_feature_names_out()[j], v * coef[j]) for j, v in zip(x.col, x.data)]
    contributions = [c for c in contributions if c[1] > 0]
    contributions.sort(key=lambda c: c[1], reverse=True)
    seen, terms = set(), []
    for term, _ in contributions:
        if term not in seen:
            seen.add(term)
            terms.append(term)
        if len(terms) == n:
            break
    return terms


class IndustryPredictor:
    """Loads the saved industry model bundle."""

    def __init__(self, path: str = INDUSTRY_MODEL_PATH):
        self.bundle = joblib.load(path) if os.path.exists(path) else None
        self.source = os.path.basename(path) if self.bundle else None

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def predict(self, sbert, text: str, k: int = 3):
        """
        Returns (ranked, unclear, terms):
          ranked: top-k (industry, probability) pairs
          unclear: True if the top probability is below UNCLEAR_BELOW
          terms: resume terms supporting the top prediction (may be empty)
        Returns None if no model is loaded.
        """
        if not self.loaded:
            return None
        emb = sbert.encode([text])
        probs = predict_proba(self.bundle, [text], emb)[0]
        classes = self.bundle["classes"]
        order = np.argsort(probs)[::-1][:k]
        ranked = [(str(classes[i]), float(probs[i])) for i in order]
        unclear = ranked[0][1] < UNCLEAR_BELOW
        terms = explain_industry(self.bundle, text, ranked[0][0])
        return ranked, unclear, terms


# =====================================================================
# Job-match scoring
# =====================================================================
# First-person, policy, or employer-voice wording: these describe the employer, not a requirement.
_NOT_A_REQUIREMENT = re.compile(
    r"\b(i|i'm|i’m|im|my|me|we|our|us|not|don't|don’t|do not|want|looking|you are|you)\b",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"^[\s\-•*·\d.)]+")

# Logistics and availability: a resume cannot show these, so they are reported separately.
_CONDITION = re.compile(
    r"\b(internet|connection|computer|laptop|hardware|equipment|hours?|schedule|shift|time zone|"
    r"timezone|standard time|location|relocat\w*|on[- ]site|remote|salary|visa|work permit|"
    r"background check)\b|\b\d{1,2}\s?(am|pm)\b",
    re.IGNORECASE,
)


# Section headings that introduce what the employer requires. Matched against short heading lines.
_REQUIREMENT_HEADINGS = re.compile(
    r"^(requirements?|qualifications?|key competenc\w*|competenc\w*|skills?|must[- ]haves?|"
    r"you (are|bring|have)|who you are|what you('ll| will)? (need|bring|have|do)|"
    r"performance objectives|responsibilities|duties|what we.re looking for|nice to have)\s*:?$",
    re.IGNORECASE,
)
# Colon-ended headings that introduce a list of requirements, e.g. 'Strong experience with the following:'.
_REQUIREMENT_WORDS = re.compile(
    r"\b(experience|proficien\w*|skills?|knowledge|qualif\w*|requir\w*|competenc\w*|ability|familiar\w*)\b",
    re.IGNORECASE,
)
# Headings for content that is not a requirement (company info, benefits, and so on).
_OTHER_HEADINGS = re.compile(
    r"^(about( the)? (company|us)|benefits|perks|compensation|company|who we are|our values|"
    r"how to apply|application process)\s*:?$",
    re.IGNORECASE,
)
_BULLET_START = re.compile(r"^\s*([-•*·]|\d+[.)])\s*")


def _is_heading(line: str, raw: str) -> bool:
    if _BULLET_START.match(raw):
        return False
    if line.endswith(":"):
        return True
    return len(line.split()) <= 4 and not line.endswith((".", "!", "?")) and (
        bool(_REQUIREMENT_HEADINGS.match(line)) or bool(_OTHER_HEADINGS.match(line))
    )


def split_requirements(job_context: str):
    """
    Splits the job description into two lists:
      competencies: skills and duties a resume can evidence
      conditions:   logistics such as hours, equipment or location

    The posting's headings decide what counts. Lines under a requirements-type heading may be
    longer and unbulleted (up to 30 words); lines under other headings or before any heading
    must be short bullet-style items (up to 14 words). Employer-voice lines are always dropped.
    """
    competencies, conditions = [], []
    in_requirements = False
    for raw in job_context.split("\n"):
        line = _BULLET_START.sub("", raw).strip()
        if not line:
            continue
        if _is_heading(line, raw):
            in_requirements = bool(_REQUIREMENT_HEADINGS.match(line)) or bool(
                _REQUIREMENT_WORDS.search(line) and line.endswith(":")
            )
            continue
        words = len(line.split())
        limit = 30 if in_requirements else 14
        fewest = 1 if in_requirements else 2
        if not (fewest <= words <= limit) or _NOT_A_REQUIREMENT.search(line):
            continue
        (conditions if _CONDITION.search(line) else competencies).append(line)
    return competencies, conditions


def requirement_lines(job_context: str) -> list:
    """All requirement lines (competencies and conditions together)."""
    competencies, conditions = split_requirements(job_context)
    return competencies + conditions


def tier_for(score: float):
    if score >= TIER_HIGH:
        return "Highly Suitable", "🟢"
    if score >= TIER_MID:
        return "Moderately Suitable", "🟡"
    return "Low Suitability", "🔴"


def calibrate(raw_hybrid: float) -> float:
    return max(0.0, min(100.0, ((raw_hybrid - CAL_LOW) / (CAL_HIGH - CAL_LOW)) * 100.0))


def calculate_suitability(sbert, resume_text: str, job_title: str, job_desc: str) -> dict:
    """Scores a resume against a job. `sbert` is a loaded SentenceTransformer."""
    from sentence_transformers import util

    job_context = f"{job_title}. {job_desc}"
    res_sentences = chunk_into_sentences(resume_text) or [resume_text]
    competencies, conditions = split_requirements(job_context)
    req_items = competencies or chunk_into_sentences(job_context) or [job_context]

    job_emb_full = sbert.encode(job_context, convert_to_tensor=True)
    res_emb_full = sbert.encode(resume_text, convert_to_tensor=True)
    res_sentence_embs = sbert.encode(res_sentences, convert_to_tensor=True)
    req_embs = sbert.encode(req_items, convert_to_tensor=True)

    base_sim = util.cos_sim(job_emb_full, res_emb_full).item()
    sims_to_jd = util.cos_sim(job_emb_full, res_sentence_embs)[0].cpu().numpy()
    top_sim = float(np.mean(np.sort(sims_to_jd)[-3:])) if len(sims_to_jd) >= 3 else base_sim

    raw_hybrid = base_sim * 0.4 + top_sim * 0.6
    score = round(calibrate(raw_hybrid), 1)

    top_indices = np.argsort(sims_to_jd)[-3:][::-1]
    strengths = []
    for i in top_indices:
        if sims_to_jd[i] > STRENGTH_MIN_SIM and res_sentences[i] not in strengths:
            strengths.append(res_sentences[i])

    cross_sims = util.cos_sim(req_embs, res_sentence_embs).cpu().numpy()
    max_sim_per_req = np.max(cross_sims, axis=1)
    weakest = np.argsort(max_sim_per_req)
    # Every listed competency, with whether the resume shows it and the sentence that does.
    best_idx = np.argmax(cross_sims, axis=1)
    listed = len(competencies) > 0
    requirements = []
    if listed:
        for i, text in enumerate(req_items):
            covered = bool(max_sim_per_req[i] >= GAP_MAX_SIM)
            requirements.append({
                "text": text,
                "covered": covered,
                "evidence": res_sentences[best_idx[i]] if covered else None,
                "similarity": float(max_sim_per_req[i]),
            })
    covered_count = sum(r["covered"] for r in requirements)
    gaps = [r["text"] for r in sorted(requirements, key=lambda r: r["similarity"]) if not r["covered"]]

    tier, color = tier_for(score)
    return {
        "score": float(score),
        "tier": tier,
        "color": color,
        "strengths": strengths,
        "weaknesses": gaps,
        "requirements": requirements,
        "req_covered": covered_count,
        "req_total": len(requirements),
        "conditions": conditions,
    }
