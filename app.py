"""
ResumeMatch AI — applicant readiness and business ranking portal.
Industry prediction: TF-IDF logistic regression trained on 2,483 labeled resumes.
Job matching: Sentence-BERT similarity, with each listed requirement checked individually.
"""

import csv
import html
import io

import streamlit as st
from sentence_transformers import SentenceTransformer

from scoring import (
    ExtractionError,
    IndustryPredictor,
    SBERT_NAME,
    calculate_suitability,
    extract_text,
)

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

st.set_page_config(page_title="ResumeMatch AI", page_icon="🌐", layout="wide")

# =====================================================================
# 1. STYLE
# Colors are neutral overlays and currentColor, so the page works in
# both Streamlit themes. Pick the theme from the ⋮ menu → Settings.
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

:root {
    --accent: #3b82f6;
    --accent-tint: rgba(59, 130, 246, 0.14);
    --ok: #16a34a;   --ok-bg: rgba(22, 163, 74, 0.12);
    --bad: #dc2626;  --bad-bg: rgba(220, 38, 38, 0.10);
    --warn: #d97706; --warn-bg: rgba(217, 119, 6, 0.12);
    --line: rgba(128, 128, 128, 0.25);
    --surface: rgba(128, 128, 128, 0.07);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 4px 12px rgba(0, 0, 0, 0.06);
}

html, body, p, li, span, label, h1, h2, h3, h4, div, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.material-symbols-rounded, [class*="material-symbols"] { font-family: 'Material Symbols Rounded' !important; }
.material-symbols-rounded {
    font-weight: normal; font-style: normal; font-size: 20px; line-height: 1;
    display: inline-block; vertical-align: middle; letter-spacing: normal;
    -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
}

.block-container { padding-top: 3.8rem; max-width: 1180px; }
p, li { line-height: 1.6; }
.stDeployButton, [data-testid="stDeployButton"] { display: none !important; }

/* Page header */
.page-head { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 8px; position: relative; }
.page-head::before { content: ""; position: absolute; left: 0; bottom: -1px; width: 56px; height: 2px; background: var(--accent); }
.page-head h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 4px 0; }
.page-head p { opacity: 0.75; margin: 0; font-size: 0.98rem; }
.eyebrow { color: var(--accent); font-size: 0.74rem; font-weight: 650; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }

/* Section headings with icon tiles */
.section { display: flex; align-items: center; gap: 10px; margin: 2.2rem 0 6px 0; }
.section .material-symbols-rounded {
    background: var(--accent-tint); color: var(--accent); width: 30px; height: 30px; border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center; font-size: 18px;
}
.section h2 { font-size: 1.05rem; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
.section-note { opacity: 0.72; font-size: 0.9rem; margin: 0 0 14px 0; }

/* Cards */
.card { border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; background: var(--surface);
        box-shadow: var(--shadow); margin-bottom: 14px; }
.label { opacity: 0.7; font-size: 0.76rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.07em; }
.big-number { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; margin-top: 4px; }
.industry-name { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 4px; }
.industry-name.small { font-size: 1.1rem; font-weight: 600; opacity: 0.8; }
.reason { line-height: 1.6; margin-top: 12px; font-size: 0.95rem; }

/* Pills */
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin: 0 6px 6px 0; border: 1px solid var(--line); }
.pill .material-symbols-rounded { font-size: 16px; }
.pill-ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; }
.pill-bad { color: var(--bad); background: var(--bad-bg); border-color: transparent; }
.pill-warn { color: var(--warn); background: var(--warn-bg); border-color: transparent; }

/* Requirement list */
.group-head { display: flex; align-items: center; gap: 8px; margin: 18px 0 8px 0; font-weight: 650; font-size: 0.92rem; }
.group-head .count { opacity: 0.6; font-weight: 500; }
.req-list { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--surface); box-shadow: var(--shadow); }
.req { display: flex; gap: 12px; padding: 13px 16px; border-top: 1px solid var(--line); }
.req:first-child { border-top: none; }
.req .material-symbols-rounded { font-size: 22px; margin-top: 1px; }
.req.covered .material-symbols-rounded { color: var(--ok); }
.req.missing .material-symbols-rounded { color: var(--bad); }
.req-text { font-weight: 550; font-size: 0.95rem; line-height: 1.45; }
.req-evidence { opacity: 0.75; font-size: 0.87rem; margin-top: 5px; line-height: 1.5; font-style: italic; }
.req-body { flex: 1; }

.condition { display: flex; gap: 12px; padding: 11px 14px; border-radius: 10px; background: var(--warn-bg);
             margin-bottom: 8px; font-size: 0.93rem; }
.condition .material-symbols-rounded { color: var(--warn); font-size: 20px; }

/* Leaderboard */
.rank { font-size: 1.6rem; font-weight: 700; opacity: 0.5; font-variant-numeric: tabular-nums; }
.candidate { font-size: 1.1rem; font-weight: 650; }
.score-sm { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; text-align: right; }

button[data-testid="stBaseButton-primary"], button[kind="primary"] {
    padding: 0.55rem 1.4rem; font-weight: 600; border-radius: 8px;
}
div[data-testid="stMetric"] { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. SIDEBAR
# =====================================================================
st.sidebar.markdown("### ResumeMatch AI")
st.sidebar.caption("Resume and job-fit analysis")
st.sidebar.divider()
selected_portal = st.sidebar.radio(
    "Portal",
    ["Applicant", "Business"],
    format_func=lambda p: "Applicant portal" if p == "Applicant" else "Business portal",
)
st.sidebar.divider()
st.sidebar.markdown("**How it works**")
st.sidebar.markdown(
    "1. The industry model reads your resume's wording.\n"
    "2. Each requirement in the posting is checked against your resume.\n"
    "3. Requirements you can't evidence are shown separately from logistics."
)
st.sidebar.divider()
st.sidebar.caption("Theme: open the ⋮ menu at the top right and choose System, Light or Dark.")


# =====================================================================
# 3. MODELS
# =====================================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return SentenceTransformer(SBERT_NAME), IndustryPredictor()


sbert_model, industry = load_models()

if industry.loaded:
    st.sidebar.caption(f"Industry model: {industry.source}")
else:
    st.sidebar.error("Industry model file not found.")


@st.cache_data(show_spinner=False)
def cached_extract(file_bytes: bytes):
    """Returns (text, error). Errors are returned rather than raised so they can be cached too."""
    try:
        return extract_text(file_bytes), None
    except ExtractionError as exc:
        return "", str(exc)


@st.cache_data(show_spinner=False)
def cached_industry(text: str):
    return industry.predict(sbert_model, text, k=3)


# =====================================================================
# 4. HTML HELPERS
# =====================================================================
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def icon(name: str) -> str:
    return f'<span class="material-symbols-rounded">{name}</span>'


def section(iname: str, title: str, note: str = "") -> None:
    st.markdown(f'<div class="section">{icon(iname)}<h2>{esc(title)}</h2></div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="section-note">{esc(note)}</p>', unsafe_allow_html=True)


def pill(text: str, kind: str = "neutral", iname: str = "") -> str:
    glyph = icon(iname) if iname else ""
    return f'<span class="pill pill-{kind}">{glyph}{esc(text)}</span>'


def _row(r) -> str:
    state = "covered" if r["covered"] else "missing"
    glyph = "check_circle" if r["covered"] else "cancel"
    evidence = ""
    if r["covered"] and r["evidence"]:
        evidence = f'<div class="req-evidence">Found in your resume: “{esc(r["evidence"])}”</div>'
    return (f'<div class="req {state}">{icon(glyph)}'
            f'<div class="req-body"><div class="req-text">{esc(r["text"])}</div>{evidence}</div></div>')


def requirements_html(requirements) -> str:
    """Not-shown requirements first, since those are what the applicant needs to act on."""
    missing = [r for r in requirements if not r["covered"]]
    covered = [r for r in requirements if r["covered"]]
    out = []
    if missing:
        out.append(f'<div class="group-head">{icon("cancel")} Not shown <span class="count">{len(missing)}</span></div>')
        out.append('<div class="req-list">' + "".join(_row(r) for r in missing) + "</div>")
    if covered:
        out.append(f'<div class="group-head">{icon("check_circle")} Covered <span class="count">{len(covered)}</span></div>')
        out.append('<div class="req-list">' + "".join(_row(r) for r in covered) + "</div>")
    return "".join(out)


def conditions_html(conditions) -> str:
    return "".join(
        f'<div class="condition">{icon("schedule")}<span>{esc(c)}</span></div>' for c in conditions
    )


def tier_kind(tier: str) -> str:
    return {"Highly Suitable": "ok", "Moderately Suitable": "warn"}.get(tier, "bad")


TIER_SENTENCES = {
    "Highly Suitable": "Most of the posting is reflected in your experience.",
    "Moderately Suitable": "Some of the posting is reflected in your experience, and the rest is missing.",
    "Low Suitability": "Most of the posting is not reflected in your experience.",
}


def describe_match(results) -> str:
    covered, total = results["req_covered"], results["req_total"]
    if total == 0:
        return ("This posting does not list individual requirements, so the score reflects "
                "the overall wording of the two documents.")
    return (f"Your resume shows evidence for <strong>{covered} of {total}</strong> listed requirements. "
            f"{TIER_SENTENCES[results['tier']]}")


def hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{esc(eyebrow)}</div>'
        f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def rejected_for_size(upload) -> bool:
    if upload.size > MAX_UPLOAD_BYTES:
        st.error(f"{upload.name} is larger than {MAX_UPLOAD_MB} MB and was skipped.")
        return True
    return False


# =====================================================================
# 5. APPLICANT PORTAL
# =====================================================================
if selected_portal == "Applicant":
    hero("Check your fit for a job",
         "See which industry your resume reads as, and which listed requirements it does and does not show.",
         "Applicant portal")

    section("upload_file", "Your resume", "PDF, up to 10 MB. Processed in server memory and not saved to disk.")
    student_file = st.file_uploader("Resume", type=["pdf"], label_visibility="collapsed")

    if student_file and not rejected_for_size(student_file):
        resume_text, error = cached_extract(student_file.getvalue())

        if error:
            st.error(f"Could not read this PDF: {error}")
        elif not industry.loaded:
            st.warning("Industry prediction is unavailable because the model file is missing.")
        else:
            prediction = cached_industry(resume_text)
            if prediction:
                ranked, unclear, terms = prediction
                label, prob = ranked[0]
                section("insights", "Likely industry")
                reason_terms = ""
                if terms:
                    reason_terms = (" The words that pushed the model most toward this result were "
                                    + ", ".join(f"<strong>{esc(t)}</strong>" for t in terms[:5]) + ".")
                if unclear:
                    headline = (f'<div class="label">Best guess</div>'
                                f'<div class="industry-name small">{esc(label)}</div>')
                    confidence_text = "The model is not confident. Treat this as a rough guess, not a conclusion."
                else:
                    headline = f'<div class="label">Predicted industry</div><div class="industry-name">{esc(label)}</div>'
                    confidence_text = "The model is reasonably confident in this result."
                alternatives = "".join(pill(f"{l} · {p:.0%}") for l, p in ranked[1:])
                st.markdown(
                    f'<div class="card">{headline}'
                    f'<div style="margin-top:10px">{pill(f"{prob:.0%} probability", "warn" if unclear else "ok")}</div>'
                    f'<div class="reason">{confidence_text}{reason_terms}</div>'
                    f'<div style="margin-top:16px" class="label">Next most likely</div>'
                    f'<div style="margin-top:6px">{alternatives}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            section("work", "Test against a job", "Enter the job title and paste the posting for a requirement-by-requirement check.")
            col_j1, col_j2 = st.columns([1, 2])
            with col_j1:
                job_title = st.text_input("Job title", placeholder="e.g., Senior Accountant")
            with col_j2:
                job_desc = st.text_area("Job posting", placeholder="Paste the full posting here...", height=140)

            if st.button("Check requirements", type="primary"):
                if not job_title.strip():
                    st.warning("Please provide a job title.")
                else:
                    with st.spinner("Checking each requirement against your resume..."):
                        results = calculate_suitability(sbert_model, resume_text, job_title, job_desc)

                    tier = results["tier"]
                    st.markdown(
                        f'<div class="card"><div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start">'
                        f'<div><div class="label">Similarity</div><div class="big-number">{results["score"]:.0f}%</div>'
                        f'<div style="margin-top:8px">{pill(results["color"] + " " + tier, tier_kind(tier))}</div></div>'
                        f'<div style="flex:1;min-width:260px"><div class="label">Requirements</div>'
                        f'<div class="big-number">{results["req_covered"]}<span style="opacity:0.6;font-size:1.4rem"> / {results["req_total"]}</span></div>'
                        f'<div class="reason">{describe_match(results)}</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                    if results["requirements"]:
                        section("checklist", "Listed requirements",
                                "Covered means a sentence in your resume closely matches the requirement. "
                                "Not shown means no sentence did.")
                        st.markdown(requirements_html(results["requirements"]), unsafe_allow_html=True)
                    else:
                        st.info("This posting does not list individual requirements to check one by one.")

                    if results["conditions"]:
                        section("schedule", "Conditions to check yourself",
                                "These are logistics. A resume cannot show them, so they are not part of the score.")
                        st.markdown(conditions_html(results["conditions"]), unsafe_allow_html=True)


# =====================================================================
# 6. BUSINESS PORTAL
# =====================================================================
else:
    hero("Rank your applicants",
         "Score every resume against the same posting, then see each candidate's requirement-by-requirement evidence.",
         "Business portal")

    col_b1, col_b2 = st.columns([1, 1], gap="large")
    with col_b1:
        section("work", "Job posting")
        biz_title = st.text_input("Job title", placeholder="e.g., Lead Financial Auditor")
        biz_desc = st.text_area("Job posting", placeholder="Paste the full posting here...", height=160)
    with col_b2:
        section("groups", "Applicant pool", "Upload PDF resumes, up to 10 MB each.")
        biz_files = st.file_uploader("Resumes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if st.button("Rank candidates", type="primary"):
        if not biz_title.strip() or not biz_files:
            st.warning("Provide a job title and upload at least one resume.")
        else:
            ranked_list, skipped = [], []
            with st.spinner("Scoring resumes..."):
                for f in biz_files:
                    if f.size > MAX_UPLOAD_BYTES:
                        skipped.append((f.name, f"larger than {MAX_UPLOAD_MB} MB"))
                        continue
                    text, error = cached_extract(f.getvalue())
                    if error:
                        skipped.append((f.name, error))
                        continue
                    metrics = calculate_suitability(sbert_model, text, biz_title, biz_desc)
                    prediction = cached_industry(text)
                    metrics.update({
                        "name": f.name.rsplit(".", 1)[0],
                        "industry": prediction[0][0][0] if prediction else None,
                    })
                    ranked_list.append(metrics)

            if skipped:
                with st.expander(f"{len(skipped)} file(s) could not be scored"):
                    for name, reason in skipped:
                        st.write(f"**{name}**: {reason}")

            if not ranked_list:
                st.error("No resumes could be scored. See the list above for the reasons.")
            else:
                ranked_list.sort(key=lambda x: x["score"], reverse=True)
                section("leaderboard", "Leaderboard", "Ranked by similarity to the posting. Open a candidate to see each requirement.")

                for rank, cand in enumerate(ranked_list, 1):
                    tier = cand["tier"]
                    st.markdown(
                        f'<div class="card" style="margin-bottom:6px"><div style="display:flex;gap:18px;align-items:center">'
                        f'<div class="rank">{rank:02d}</div>'
                        f'<div style="flex:1"><div class="candidate">{esc(cand["name"])}</div>'
                        f'<div style="margin-top:6px">{pill(cand["color"] + " " + tier, tier_kind(tier))}'
                        f'{pill("Looks like " + cand["industry"]) if cand["industry"] else ""}</div></div>'
                        f'<div><div class="score-sm">{cand["score"]:.0f}%</div>'
                        f'<div class="label" style="text-align:right">{cand["req_covered"]} / {cand["req_total"]} requirements</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Requirement-by-requirement breakdown"):
                        st.markdown(f'<div class="reason" style="margin:0 0 12px 0">{describe_match(cand)}</div>', unsafe_allow_html=True)
                        if cand["requirements"]:
                            st.markdown(requirements_html(cand["requirements"]), unsafe_allow_html=True)
                        if cand["conditions"]:
                            st.markdown("**Conditions to check yourself**")
                            st.markdown(conditions_html(cand["conditions"]), unsafe_allow_html=True)

                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerow(["rank", "name", "similarity_score", "tier", "industry",
                                 "requirements_covered", "requirements_total", "not_shown_requirements"])
                for rank, cand in enumerate(ranked_list, 1):
                    writer.writerow([
                        rank, cand["name"], cand["score"], cand["tier"], cand["industry"] or "",
                        cand["req_covered"], cand["req_total"], "; ".join(cand["weaknesses"]),
                    ])
                st.download_button(
                    "Download leaderboard (CSV)",
                    data=buffer.getvalue(),
                    file_name="candidate_leaderboard.csv",
                    mime="text/csv",
                )
