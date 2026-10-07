"""
Matching-quality check without human ratings.

Ground truth = the industry folder a resume came from. For each resume we rank 24
reference job descriptions (one per industry) by similarity and ask: does the resume's
own industry rank first? This tests whether the matching signal points the right way.

Note: this measures the base document similarity, which is one component of the app's
score. It does not validate the score's tier thresholds.
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from train_industry_classifier import load_dataset
import scoring

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "matching_evaluation.json")

# One reference job description per industry, written to describe typical duties.
JOB_DESCRIPTIONS = {
    "ACCOUNTANT": "Prepare financial statements, reconcile ledgers, manage accounts payable and receivable, file tax returns and support month-end and year-end close.",
    "ADVOCATE": "Represent clients in court, draft legal pleadings and contracts, conduct legal research, advise on litigation strategy and maintain case files.",
    "AGRICULTURE": "Manage crop production and livestock, apply soil and irrigation practices, monitor pests, operate farm machinery and track yields.",
    "APPAREL": "Design and produce garments, source fabrics, manage merchandising and sizing, coordinate production with suppliers and track retail sales.",
    "ARTS": "Create original artwork, curate exhibitions, prepare portfolios, teach studio classes and collaborate with galleries and cultural organizations.",
    "AUTOMOBILE": "Diagnose and repair vehicles, perform engine and brake service, read diagnostic equipment, maintain service records and advise customers.",
    "AVIATION": "Operate aircraft or manage flight operations, follow aviation safety regulations, perform pre-flight checks, plan routes and maintain flight logs.",
    "BANKING": "Manage customer accounts, process loans and credit applications, assess credit risk, ensure regulatory compliance and handle branch operations.",
    "BPO": "Handle inbound and outbound customer calls, resolve support tickets, use CRM systems, meet service level targets and maintain quality scores.",
    "BUSINESS-DEVELOPMENT": "Identify new markets and partnerships, generate qualified leads, negotiate deals, build client relationships and track pipeline targets.",
    "CHEF": "Plan menus, prepare and cook dishes, manage kitchen inventory and food costs, supervise kitchen staff and maintain food safety standards.",
    "CONSTRUCTION": "Supervise construction sites, read blueprints, manage contractors and schedules, ensure safety compliance and control materials and budgets.",
    "CONSULTANT": "Analyze client problems, design recommendations, lead workshops, produce strategy reports and manage project deliverables for clients.",
    "DESIGNER": "Create visual designs and prototypes, use Adobe Creative Suite or Figma, develop brand identities and collaborate with clients on design direction.",
    "DIGITAL-MEDIA": "Produce digital content, manage social media channels, create video and graphics, run online campaigns and analyze audience engagement metrics.",
    "ENGINEERING": "Design and test mechanical or electrical systems, use CAD software, prepare technical specifications, and solve engineering problems in production.",
    "FINANCE": "Build financial models and forecasts, analyze investments, prepare budgets and reports, manage risk and advise leadership on financial decisions.",
    "FITNESS": "Design training programs, lead group and personal training sessions, assess client fitness, promote healthy habits and maintain gym safety standards.",
    "HEALTHCARE": "Provide patient care, administer medications, record clinical notes, coordinate with physicians and follow infection control and medical protocols.",
    "HR": "Recruit and onboard employees, manage payroll and benefits, handle employee relations, administer performance reviews and ensure labor law compliance.",
    "INFORMATION-TECHNOLOGY": "Develop and maintain software systems, manage networks and servers, write Python or Java code, support users and secure company infrastructure.",
    "PUBLIC-RELATIONS": "Write press releases, manage media relations, plan communication campaigns, prepare executive messaging and monitor brand reputation.",
    "SALES": "Meet monthly revenue quotas, prospect and close deals, manage client accounts, deliver product presentations and maintain CRM records.",
    "TEACHER": "Plan lessons and curricula, teach students in classrooms, assess learning progress, communicate with parents and manage classroom behavior.",
}


def embed_resumes(texts, names, model_name):
    """Uses the cached MiniLM embeddings when available, otherwise encodes with the given model."""
    if model_name == scoring.SBERT_NAME and os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH, allow_pickle=True)
        return cache["embeddings"]
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def evaluate(scores, labels, classes):
    """scores: (n_resumes, n_classes). Returns top-1, top-3, and mean reciprocal rank."""
    class_index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([class_index[l] for l in labels])
    order = np.argsort(-scores, axis=1)
    ranks = np.array([np.where(order[i] == true_idx[i])[0][0] + 1 for i in range(len(labels))])
    return {
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "mrr": float((1.0 / ranks).mean()),
    }


def main():
    texts, labels, names = load_dataset()
    classes = sorted(JOB_DESCRIPTIONS)
    jd_texts = [JOB_DESCRIPTIONS[c] for c in classes]
    labels = np.array(labels)

    results = {}

    # 1) TF-IDF cosine (keyword baseline)
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=2)
    vec.fit(list(texts) + jd_texts)
    res_tfidf = vec.transform(texts)
    jd_tfidf = vec.transform(jd_texts)
    tfidf_scores = (res_tfidf @ jd_tfidf.T).toarray()
    results["tfidf"] = evaluate(tfidf_scores, labels, classes)

    # 2) Sentence-BERT cosine (MiniLM, the model the app uses)
    minilm = SentenceTransformer(scoring.SBERT_NAME)
    res_emb = embed_resumes(texts, names, scoring.SBERT_NAME)
    jd_emb = minilm.encode(jd_texts)
    from sklearn.preprocessing import normalize
    minilm_scores = normalize(res_emb) @ normalize(jd_emb).T
    results[scoring.SBERT_NAME] = evaluate(minilm_scores, labels, classes)

    # 3) A stronger sentence model, if it can be downloaded
    alt = "all-mpnet-base-v2"
    try:
        alt_model = SentenceTransformer(alt)
        alt_res = alt_model.encode(texts, show_progress_bar=True)
        alt_jd = alt_model.encode(jd_texts)
        results[alt] = evaluate(normalize(alt_res) @ normalize(alt_jd).T, labels, classes)
    except Exception as exc:
        print(f"Skipped {alt}: {exc}")

    print(f"{'method':<28}{'top1':>8}{'top3':>8}{'MRR':>8}")
    for name, r in results.items():
        print(f"{name:<28}{r['top1']:>8.3f}{r['top3']:>8.3f}{r['mrr']:>8.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
