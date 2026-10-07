import io

import pytest
from pypdf import PdfWriter

import scoring
from scoring import ExtractionError, calibrate, chunk_into_sentences, tier_for


def make_pdf_bytes(text: str) -> bytes:
    """Builds a one-page PDF containing `text` as extractable content."""
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]

    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------- pure functions: no models needed ----------

def test_calibrate_clamps_to_0_100():
    assert calibrate(0.0) == 0.0
    assert calibrate(1.0) == 100.0


def test_calibrate_midpoint():
    # raw 0.45 is the midpoint of the 0.20-0.70 range
    assert calibrate(0.45) == pytest.approx(50.0)


@pytest.mark.parametrize(
    "score, tier",
    [(80, "Highly Suitable"), (75, "Highly Suitable"), (60, "Moderately Suitable"), (10, "Low Suitability")],
)
def test_tier_thresholds(score, tier):
    assert tier_for(score)[0] == tier


def test_chunk_drops_short_fragments():
    text = "Short. This sentence has more than three words. Tiny one."
    chunks = chunk_into_sentences(text)
    assert chunks == ["This sentence has more than three words."]


def test_chunk_handles_bullets():
    text = "• Managed a team of five engineers • Delivered three large projects"
    chunks = chunk_into_sentences(text)
    assert all(len(c.split()) > 3 for c in chunks)
    assert len(chunks) >= 1


def test_requirement_lines_skip_employer_voice():
    jd = (
        "I'm looking for a highly creative assistant to work with me.\n"
        "NOT LOOKING FOR AI CREATIVES.\n"
        "This is not a traditional virtual assistant role.\n"
        "- Adobe Photoshop\n"
        "- Photo editing and color correction\n"
        "- Typography\n"
    )
    items = scoring.requirement_lines(jd)
    assert "Adobe Photoshop" in items
    assert "Photo editing and color correction" in items
    assert all("looking" not in i.lower() and "not" not in i.lower().split() for i in items)


def test_extract_text_rejects_garbage_bytes():
    with pytest.raises(ExtractionError):
        scoring.extract_text(b"not a pdf at all")


def test_extract_text_rejects_empty_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ExtractionError):
        scoring.extract_text(buf.getvalue())


def test_extract_text_reads_real_text():
    body = "Experienced accountant with ten years of audit and tax preparation experience in regional firms."
    text = scoring.extract_text(make_pdf_bytes(body))
    assert "accountant" in text.lower()


# ---------- integration: uses the real Sentence-BERT model ----------

@pytest.fixture(scope="module")
def sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(scoring.SBERT_NAME)


def test_relevant_resume_outscores_unrelated(sbert):
    relevant = (
        "Senior accountant with eight years of experience in financial audit, tax filing, "
        "ledger reconciliation and month-end close for multinational companies."
    )
    unrelated = (
        "Passionate pastry chef who trained in Paris and runs a bakery serving fresh bread, "
        "croissants and seasonal desserts every morning."
    )
    job = "Senior Accountant. Prepare financial statements, manage audits, reconcile ledgers and file corporate tax returns."
    good = scoring.calculate_suitability(sbert, relevant, "Senior Accountant", job)
    bad = scoring.calculate_suitability(sbert, unrelated, "Senior Accountant", job)
    assert good["score"] > bad["score"]
    assert good["tier"] in {"Highly Suitable", "Moderately Suitable", "Low Suitability"}


def test_suitability_output_shape(sbert):
    result = scoring.calculate_suitability(sbert, "Some resume text about software engineering work and projects.", "Developer", "Build web apps.")
    assert set(result) == {"score", "tier", "color", "strengths", "weaknesses", "requirements", "req_covered", "req_total", "conditions"}
    assert 0.0 <= result["score"] <= 100.0
    assert len(result["strengths"]) <= 3
    assert len(result["weaknesses"]) <= 3


def test_industry_predictor_returns_ranked_top3(sbert):
    predictor = scoring.IndustryPredictor()
    assert predictor.loaded, "industry model file should be present"
    result = predictor.predict(sbert, "Certified public accountant with audit and tax experience in finance teams.", k=3)
    ranked, unclear, terms = result
    assert len(ranked) == 3
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-6
    assert isinstance(unclear, bool)
    assert isinstance(terms, list)


def test_industry_predictor_disabled_when_missing(tmp_path):
    predictor = scoring.IndustryPredictor(path=str(tmp_path / "missing.joblib"))
    assert not predictor.loaded
    assert predictor.predict(None, "anything") is None
