# ResumeMatch AI

An applicant-readiness and candidate-ranking tool. Upload a resume to get an industry
prediction and a requirement-by-requirement check against a job posting, or upload a pool
of resumes to rank them against the same posting.

## How it works

- **Industry prediction**: a TF-IDF + logistic regression classifier trained on 2,483
  labeled resumes across 24 industries (`compare_classifiers.py`, `train_industry_classifier.py`).
- **Job matching**: Sentence-BERT (`all-MiniLM-L6-v2`) embeddings compare the resume against
  the job posting as a whole and against each listed requirement individually.
- **Requirement checking**: the posting is split into competencies (skills a resume can
  evidence) and conditions (logistics like hours or equipment, which can't be evidenced).

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

- `app.py` — the Streamlit UI
- `scoring.py` — scoring and classification logic, UI-independent
- `test_scoring.py` — unit and integration tests (`pytest -q`)
- `train_industry_classifier.py`, `compare_classifiers.py` — retrain/compare the classifier
- `evaluate_matching.py` — checks the matching logic against industry labels as a proxy
  for ground truth
- `industry_model_v3.joblib` — the deployed classifier bundle

## Notes

- The resume dataset used for training is not included in this repository.
- `training_report.txt`, `classifier_comparison.json`, and `matching_evaluation.json`
  record the evaluation results referenced during development.
