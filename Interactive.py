"""
PortfolioMatch AI — Student Self-Assessment Web Portal
Features:
  1. Auto-Match / Role Discovery Mode (Scans resume against all 6 roles to find top fits)
  2. Targeted Role Mode (Evaluates readiness for one specific role)
Run with: streamlit run app.py
"""

import io
import re
from typing import Dict, List, Set, Tuple, Any
import pypdf
import streamlit as st

# ==========================================
# 1. STANDARDIZED TAXONOMY & MAPPINGS
# ==========================================

ROLE_SKILL_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Web Developer": {
        "core": ["HTML", "CSS", "JavaScript", "React", "Responsive Design", "Git"],
        "supporting": ["TypeScript", "Tailwind CSS", "Bootstrap", "REST API", "Node.js", "Webpack"]
    },
    "Mobile Developer": {
        "core": ["Kotlin", "Swift", "Flutter", "React Native", "Android SDK", "iOS SDK"],
        "supporting": ["Dart", "Java", "Firebase", "REST API", "Mobile UI", "Git"]
    },
    "UI/UX Designer": {
        "core": ["Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research", "Usability Testing"],
        "supporting": ["Design Systems", "Information Architecture", "HTML", "CSS", "Interaction Design", "Photoshop"]
    },
    "Backend Developer": {
        "core": ["Node.js", "Python", "SQL", "REST API", "PostgreSQL", "MongoDB"],
        "supporting": ["Docker", "Redis", "Express.js", "Django", "FastAPI", "Git", "AWS"]
    },
    "Full Stack Developer": {
        "core": ["JavaScript", "React", "Node.js", "SQL", "REST API", "Git"],
        "supporting": ["TypeScript", "HTML", "CSS", "Express.js", "PostgreSQL", "MongoDB", "Docker", "AWS"]
    },
    "Data/ML Developer": {
        "core": ["Python", "SQL", "Pandas", "Scikit-Learn", "Machine Learning", "Data Analysis"],
        "supporting": ["NumPy", "TensorFlow", "PyTorch", "Data Visualization", "Matplotlib", "Jupyter", "Git"]
    }
}

SKILL_CANONICAL_LOOKUP: Dict[str, str] = {
    "html": "HTML", "html5": "HTML",
    "css": "CSS", "css3": "CSS",
    "javascript": "JavaScript", "js": "JavaScript", "ecmascript": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "python": "Python", "python3": "Python",
    "java": "Java", "dart": "Dart", "kotlin": "Kotlin", "swift": "Swift",
    "react": "React", "react.js": "React", "reactjs": "React",
    "vue": "Vue.js", "vue.js": "Vue.js", "vuejs": "Vue.js",
    "tailwind": "Tailwind CSS", "tailwind css": "Tailwind CSS", "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap", "bootstrap 5": "Bootstrap",
    "responsive design": "Responsive Design", "responsive web design": "Responsive Design",
    "webpack": "Webpack",
    "flutter": "Flutter", "react native": "React Native", "react-native": "React Native",
    "android sdk": "Android SDK", "android development": "Android SDK", "android": "Android SDK",
    "ios sdk": "iOS SDK", "ios development": "iOS SDK", "ios": "iOS SDK",
    "mobile ui": "Mobile UI", "mobile ui design": "Mobile UI",
    "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js",
    "express": "Express.js", "express.js": "Express.js", "expressjs": "Express.js",
    "django": "Django", "fastapi": "FastAPI",
    "sql": "SQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mysql": "MySQL", "mongodb": "MongoDB", "mongo": "MongoDB",
    "redis": "Redis", "rest api": "REST API", "restful api": "REST API", "rest apis": "REST API", "rest": "REST API",
    "pandas": "Pandas", "numpy": "NumPy",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "data analysis": "Data Analysis", "data analytics": "Data Analysis",
    "data visualization": "Data Visualization", "matplotlib": "Matplotlib", "seaborn": "Data Visualization",
    "jupyter": "Jupyter", "jupyter notebook": "Jupyter",
    "figma": "Figma", "adobe xd": "Adobe XD", "xd": "Adobe XD",
    "wireframing": "Wireframing", "wireframes": "Wireframing", "wireframe": "Wireframing",
    "prototyping": "Prototyping", "prototype": "Prototyping", "prototypes": "Prototyping",
    "user research": "User Research", "usability testing": "Usability Testing",
    "design systems": "Design Systems", "design system": "Design Systems",
    "information architecture": "Information Architecture", "photoshop": "Photoshop", "interaction design": "Interaction Design",
    "git": "Git", "github": "Git", "version control": "Git",
    "docker": "Docker", "aws": "AWS", "amazon web services": "AWS",
    "firebase": "Firebase"
}


# ==========================================
# 2. CORE EXTRACTION & SCORING FUNCTIONS
# ==========================================

def extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_skills(text: str) -> Set[str]:
    found = set()
    clean = text.lower()
    for pattern in sorted(SKILL_CANONICAL_LOOKUP.keys(), key=len, reverse=True):
        regex = rf"(?:\b|\A){re.escape(pattern)}(?:\b|\Z)"
        if re.search(regex, clean):
            found.add(SKILL_CANONICAL_LOOKUP[pattern])
    return found


def extract_project_depth(text: str) -> Tuple[int, float]:
    lines = text.split("\n")
    count = 0
    in_proj = False
    for line in lines:
        lower = line.strip().lower()
        if "project" in lower and len(lower) < 35:
            in_proj = True
            continue
        if in_proj and any(h in lower for h in ["education", "experience", "certifications", "skills"]):
            in_proj = False
        if in_proj and (line.strip().startswith(("-", "•")) or "http" in lower or "github.com" in lower):
            count += 1

    if count == 0:
        count = len(re.findall(r"(?:project\s*\d+|github\.com\/|built\s+a|developed\s+a)", text, re.IGNORECASE))

    multiplier = 0.25 if count == 0 else (0.55 if count == 1 else (0.80 if count == 2 else 1.00))
    return count, multiplier


def compute_role_compatibility(role_name: str, extracted_skills: Set[str], proj_count: int, depth_score: float) -> Dict[str, Any]:
    profile = ROLE_SKILL_TAXONOMY[role_name]
    core_reqs = set(profile["core"])
    supp_reqs = set(profile["supporting"])

    matched_core = sorted(list(extracted_skills.intersection(core_reqs)))
    missing_core = sorted(list(core_reqs.difference(extracted_skills)))
    matched_supp = sorted(list(extracted_skills.intersection(supp_reqs)))
    missing_supp = sorted(list(supp_reqs.difference(extracted_skills)))

    core_pct = len(matched_core) / len(core_reqs)
    supp_pct = len(matched_supp) / len(supp_reqs)

    score = round(((core_pct * 0.60) + (supp_pct * 0.25) + (depth_score * 0.15)) * 100.0, 1)

    if score >= 75.0:
        tier = "Strong Fit (Ready to Bid)"
        badge_color = "🟢"
        summary = f"Your resume demonstrates strong coverage of the core requirements for {role_name}."
        advice = f"Consider highlighting experience with **{', '.join(missing_core)}** to reach full core coverage." if missing_core else "Your core tech stack is complete. You are in a prime position to apply."
    elif score >= 45.0:
        tier = "Developing Fit (Solid Foundation)"
        badge_color = "🟡"
        summary = f"You have a working foundation for {role_name}, but building a few core skills will increase your selection rate."
        advice = f"Focus on building 1 project showcasing **{' and '.join(missing_core[:2])}** before bidding."
    else:
        tier = "Early Stage (Skill Building)"
        badge_color = "🔴"
        summary = f"Your profile currently has limited overlap with the standard {role_name} stack."
        advice = f"Follow a learning pathway covering **{', '.join(missing_core[:3])}**."

    return {
        "role": role_name,
        "score": score,
        "tier": tier,
        "badge_color": badge_color,
        "summary": summary,
        "advice": advice,
        "matched_core": matched_core,
        "missing_core": missing_core,
        "matched_supp": matched_supp,
        "missing_supp": missing_supp,
        "core_count_str": f"{len(matched_core)}/{len(core_reqs)}",
        "core_pct": round(core_pct * 100),
        "supp_count_str": f"{len(matched_supp)}/{len(supp_reqs)}",
        "supp_pct": round(supp_pct * 100),
        "proj_count": proj_count,
        "depth_score": depth_score
    }


# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="PortfolioMatch AI | Student Readiness Hub", page_icon="🧭", layout="wide")

st.title("🧭 PortfolioMatch AI — Student Readiness & Role Discovery")
st.markdown("Upload your resume to discover which freelance developer roles best match your skill set, or check your readiness for a specific role.")

st.divider()

# Mode Selection & Upload Area
col_mode, col_upload = st.columns([1, 1])

with col_mode:
    st.subheader("Step 1: Choose Evaluation Mode")
    eval_mode = st.radio(
        "How would you like to assess your resume?",
        options=[
            "🌟 Find My Best Fit (Scan All 6 Roles)",
            "🎯 Test Specific Role (Targeted Check)"
        ],
        index=0
    )

    selected_role = None
    if eval_mode == "🎯 Test Specific Role (Targeted Check)":
        selected_role = st.selectbox(
            "Select the role to target:",
            options=list(ROLE_SKILL_TAXONOMY.keys())
        )

with col_upload:
    st.subheader("Step 2: Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])

# Execution & Display
if uploaded_file is not None:
    try:
        raw_text = extract_pdf_text_from_bytes(uploaded_file.read())
        
        if not raw_text.strip():
            st.error("⚠️ Could not extract text from this PDF. Please verify that your document has a readable text layer.")
        else:
            extracted_skills = extract_skills(raw_text)
            proj_count, depth_score = extract_project_depth(raw_text)

            # Global Skill Bar
            with st.expander("🔎 View All Detected Skills from Your Resume", expanded=False):
                if extracted_skills:
                    st.write(" ".join([f"`{s}`" for s in sorted(list(extracted_skills))]))
                    st.caption(f"Total skills identified: {len(extracted_skills)} | Project markers detected: {proj_count}")
                else:
                    st.write("No recognized technical skills found in the skill dictionary.")

            st.divider()

            # -------------------------------------------------------------
            # MODE A: ALL ROLES DISCOVERY
            # -------------------------------------------------------------
            if eval_mode == "🌟 Find My Best Fit (Scan All 6 Roles)":
                all_evals = []
                for role in ROLE_SKILL_TAXONOMY.keys():
                    res = compute_role_compatibility(role, extracted_skills, proj_count, depth_score)
                    all_evals.append(res)

                # Sort by score descending
                all_evals.sort(key=lambda x: x["score"], reverse=True)
                top_role = all_evals[0]

                # Top Recommendation Hero Card
                st.subheader("🏆 Your Top Role Match")
                hero_col1, hero_col2 = st.columns([1, 2])
                
                with hero_col1:
                    st.metric(
                        label=f"Highest Fit: {top_role['role']}",
                        value=f"{top_role['score']}%",
                        delta=f"{top_role['tier']}"
                    )
                    st.progress(top_role["score"] / 100.0)

                with hero_col2:
                    st.write(f"**Readiness Overview:** {top_role['summary']}")
                    st.info(f"💡 **Actionable Step:** {top_role['advice']}")

                st.markdown("### 📊 Role Compatibility Leaderboard")
                st.caption("Here is how your resume scores across all six freelance categories:")

                # Leaderboard Summary Cards
                for idx, ev in enumerate(all_evals, 1):
                    with st.container():
                        card_c1, card_c2, card_c3, card_c4 = st.columns([2, 1, 1, 4])
                        card_c1.markdown(f"**#{idx} {ev['role']}**")
                        card_c2.markdown(f"**{ev['score']}%** ({ev['badge_color']})")
                        card_c3.markdown(f"Core: `{ev['core_count_str']}`")
                        card_c4.caption(f"{ev['tier']} — Missing: {', '.join(ev['missing_core'][:2]) if ev['missing_core'] else 'None'}")
                        st.progress(ev["score"] / 100.0)

                # Detailed Accordion Breakdowns
                st.markdown("---")
                st.subheader("🔬 Deep-Dive Skill Gaps by Role")
                for ev in all_evals:
                    with st.expander(f"{ev['badge_color']} {ev['role']} — {ev['score']}% Match Breakdown"):
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            st.markdown(f"**Core Skills ({ev['core_count_str']} matched):**")
                            st.write("Matched: " + (" ".join([f"`{s}`" for s in ev['matched_core']]) if ev['matched_core'] else "_None_"))
                            st.write("Missing: " + (" ".join([f"`{s}`" for s in ev['missing_core']]) if ev['missing_core'] else "🎉 _All core skills present!_"))
                        with d_col2:
                            st.markdown(f"**Supporting Skills ({ev['supp_count_str']} matched):**")
                            st.write("Matched: " + (" ".join([f"`{s}`" for s in ev['matched_supp']]) if ev['matched_supp'] else "_None_"))
                            st.write("Missing: " + (" ".join([f"`{s}`" for s in ev['missing_supp']]) if ev['missing_supp'] else "🎉 _All supporting skills present!_"))
                        st.info(f"💡 **Next step for this role:** {ev['advice']}")

            # -------------------------------------------------------------
            # MODE B: TARGETED ROLE CHECK
            # -------------------------------------------------------------
            else:
                result = compute_role_compatibility(selected_role, extracted_skills, proj_count, depth_score)
                
                st.subheader(f"📊 Compatibility Breakdown for: {selected_role}")
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Match Score", f"{result['score']}%")
                kpi2.metric("Core Coverage", result["core_count_str"], f"{result['core_pct']}%")
                kpi3.metric("Supporting Coverage", result["supp_count_str"], f"{result['supp_pct']}%")
                kpi4.metric("Projects Found", f"{result['proj_count']} items", f"{result['depth_score']}x weight")

                st.progress(result["score"] / 100.0)

                if result["score"] >= 75.0:
                    st.success(f"### {result['badge_color']} {result['tier']}")
                elif result["score"] >= 45.0:
                    st.warning(f"### {result['badge_color']} {result['tier']}")
                else:
                    st.error(f"### {result['badge_color']} {result['tier']}")

                st.write(result["summary"])
                st.info(f"💡 **Actionable Next Step:** {result['advice']}")

                st.divider()
                st.subheader("🔍 Skill Comparison")
                c_core, c_supp = st.columns(2)

                with c_core:
                    st.markdown("#### Core Skills (60% Weight)")
                    st.markdown("**Matched:**")
                    st.write(" ".join([f"`{s}`" for s in result["matched_core"]]) if result["matched_core"] else "*None detected*")
                    st.markdown("**Missing / To Learn:**")
                    st.write(" ".join([f"`{s}`" for s in result["missing_core"]]) if result["missing_core"] else "🎉 *100% Core coverage reached!*")

                with c_supp:
                    st.markdown("#### Supporting Skills (25% Weight)")
                    st.markdown("**Matched:**")
                    st.write(" ".join([f"`{s}`" for s in result["matched_supp"]]) if result["matched_supp"] else "*None detected*")
                    st.markdown("**Unmatched Supporting:**")
                    st.write(" ".join([f"`{s}`" for s in result["missing_supp"]]) if result["missing_supp"] else "🎉 *All supporting skills matched!*")

    except Exception as e:
        st.error(f"Error analyzing PDF: {e}")

else:
    st.info("👆 Select an assessment mode and upload your PDF resume above to begin.")