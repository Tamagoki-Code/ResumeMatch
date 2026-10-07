"""
PortfolioMatch AI — Student Self-Assessment Module
Component: Single-Resume Compatibility Scoring & Actionable Feedback Engine
"""

import os
import re
import math
from typing import Dict, List, Set, Tuple, Any, Optional

# PDF text extraction with multi-library fallback
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF file using available PDF libraries
    (PyMuPDF / fitz, pdfplumber, or pypdf/PyPDF2).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at path: {pdf_path}")
        
    extracted_text = ""
    
    # Attempt 1: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            extracted_text += page.get_text() + "\n"
        if extracted_text.strip():
            return extracted_text
    except ImportError:
        pass

    # Attempt 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        if extracted_text.strip():
            return extracted_text
    except ImportError:
        pass

    # Attempt 3: pypdf / PyPDF2
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        if extracted_text.strip():
            return extracted_text
    except ImportError:
        pass

    if not extracted_text.strip():
        # Fallback for plain text mock files during local test runs
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        except Exception:
            pass

    return extracted_text


# ==========================================
# 1. STANDARDIZED ROLE TAXONOMY & ALIAS MAP
# Grounded in O*NET and roadmap.sh
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
    # Web & Core Languages
    "html": "HTML", "html5": "HTML",
    "css": "CSS", "css3": "CSS",
    "javascript": "JavaScript", "js": "JavaScript", "ecmascript": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "python": "Python", "python3": "Python",
    "java": "Java", "dart": "Dart", "kotlin": "Kotlin", "swift": "Swift",
    
    # Frontend Libraries & Styling
    "react": "React", "react.js": "React", "reactjs": "React",
    "vue": "Vue.js", "vue.js": "Vue.js", "vuejs": "Vue.js",
    "tailwind": "Tailwind CSS", "tailwind css": "Tailwind CSS", "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap", "bootstrap 5": "Bootstrap",
    "responsive design": "Responsive Design", "responsive web design": "Responsive Design",
    "webpack": "Webpack",
    
    # Mobile
    "flutter": "Flutter", "react native": "React Native", "react-native": "React Native",
    "android sdk": "Android SDK", "android development": "Android SDK", "android": "Android SDK",
    "ios sdk": "iOS SDK", "ios development": "iOS SDK", "ios": "iOS SDK",
    "mobile ui": "Mobile UI", "mobile ui design": "Mobile UI",
    
    # Backend & Databases
    "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js",
    "express": "Express.js", "express.js": "Express.js", "expressjs": "Express.js",
    "django": "Django", "fastapi": "FastAPI",
    "sql": "SQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mysql": "MySQL", "mongodb": "MongoDB", "mongo": "MongoDB",
    "redis": "Redis", "rest api": "REST API", "restful api": "REST API", "rest apis": "REST API", "rest": "REST API",
    
    # Data & Machine Learning
    "pandas": "Pandas", "numpy": "NumPy",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "data analysis": "Data Analysis", "data analytics": "Data Analysis",
    "data visualization": "Data Visualization", "matplotlib": "Matplotlib", "seaborn": "Data Visualization",
    "jupyter": "Jupyter", "jupyter notebook": "Jupyter",
    
    # UI/UX & Design
    "figma": "Figma", "adobe xd": "Adobe XD", "xd": "Adobe XD",
    "wireframing": "Wireframing", "wireframes": "Wireframing", "wireframe": "Wireframing",
    "prototyping": "Prototyping", "prototype": "Prototyping", "prototypes": "Prototyping",
    "user research": "User Research", "usability testing": "Usability Testing",
    "design systems": "Design Systems", "design system": "Design Systems",
    "information architecture": "Information Architecture", "photoshop": "Photoshop", "interaction design": "Interaction Design",
    
    # Cloud, DevOps & Tools
    "git": "Git", "github": "Git", "version control": "Git",
    "docker": "Docker", "aws": "AWS", "amazon web services": "AWS",
    "firebase": "Firebase"
}


# ==========================================
# 2. SELF-ASSESSMENT COMPATIBILITY ENGINE
# ==========================================

class StudentSelfAssessmentEngine:
    def __init__(self, core_weight: float = 0.60, supp_weight: float = 0.25, depth_weight: float = 0.15):
        self.w_core = core_weight
        self.w_supp = supp_weight
        self.w_depth = depth_weight

    def extract_skills(self, text: str) -> Set[str]:
        """Extracts and canonicalizes skill entities using regex boundary parsing."""
        if not text or not text.strip():
            return set()
            
        found_skills: Set[str] = set()
        clean_text = text.lower()
        
        # Sort keys descending to prioritize multi-word entities (e.g. "React Native" over "React")
        sorted_patterns = sorted(SKILL_CANONICAL_LOOKUP.keys(), key=len, reverse=True)
        
        for pattern in sorted_patterns:
            escaped_pat = re.escape(pattern)
            regex = rf"(?:\b|\A){escaped_pat}(?:\b|\Z)"
            if re.search(regex, clean_text):
                canonical_name = SKILL_CANONICAL_LOOKUP[pattern]
                found_skills.add(canonical_name)
                
        return found_skills

    def evaluate_project_depth(self, text: str) -> Tuple[int, float]:
        """
        Parses project mentions as a proxy for portfolio/code verification.
        Returns (project_count, depth_score [0.25 - 1.00]).
        """
        if not text or not text.strip():
            return 0, 0.25

        # Heuristic 1: Scan for project headers, bullet indicators, repository links
        lines = text.split("\n")
        project_count = 0
        in_project_section = False
        
        for line in lines:
            lower = line.strip().lower()
            if "project" in lower and len(lower) < 35:
                in_project_section = True
                continue
            if in_project_section and any(h in lower for h in ["education", "experience", "work history", "certifications", "skills"]):
                in_project_section = False
                
            if in_project_section and (line.strip().startswith("-") or line.strip().startswith("•") or "github.com" in lower or "http" in lower):
                project_count += 1

        # Heuristic 2: Keyword pattern scanning fallback
        if project_count == 0:
            matches = re.findall(r"(?:project\s*\d+|github\.com\/|built\s+a|developed\s+a|created\s+a)", text, re.IGNORECASE)
            project_count = len(matches)

        # Normalization mapping
        if project_count == 0:
            depth_score = 0.25
        elif project_count == 1:
            depth_score = 0.55
        elif project_count == 2:
            depth_score = 0.80
        else:
            depth_score = 1.00
            
        return project_count, depth_score

    def generate_guidance(self, score: float, missing_core: List[str], missing_supp: List[str], proj_count: int, role: str) -> Dict[str, Any]:
        """Generates constructive, student-centered readiness verdicts and actionable steps."""
        if score >= 75.0:
            verdict_badge = "Strong Fit (Ready to Bid)"
            readiness_summary = (
                f"You have a competitive skill profile for {role} listings! Your resume demonstrates "
                "the core proficiencies local MSMEs look for when selecting student developers."
            )
            if missing_core:
                actionable_advice = (
                    f"To make your profile stand out even more, consider highlighting experience with "
                    f"{', '.join(missing_core)} in your summary or project descriptions."
                )
            elif proj_count < 2:
                actionable_advice = "Your core tech stack is complete. Adding one more live project link will maximize your interview callback rate."
            else:
                actionable_advice = "Your profile is primed for client listings. Start browsing open gigs and submit your bids with confidence!"

        elif score >= 45.0:
            verdict_badge = "Developing Fit (Solid Foundation)"
            readiness_summary = (
                f"You have a solid technical foundation for {role}, but closing a few core skill gaps "
                "will substantially increase your chances of being selected by business owners."
            )
            if missing_core:
                top_priority = missing_core[:2]
                actionable_advice = (
                    f"Priority improvement: Build or showcase a small capstone project featuring "
                    f"{' and '.join(top_priority)} before bidding on higher-budget client listings."
                )
            else:
                actionable_advice = "Focus on expanding your supporting skills and linking active GitHub repositories to your application."

        else:
            verdict_badge = "Early Stage (Skill Building Recommended)"
            readiness_summary = (
                f"Your resume currently matches few of the standard core requirements for {role}. "
                "Local MSMEs prioritize applicants who can hit the ground running with core tools."
            )
            if missing_core:
                first_steps = missing_core[:3]
                actionable_advice = (
                    f"Recommended starting point: Follow a structured roadmap focusing on {', '.join(first_steps)}. "
                    "Complete 1–2 practical demo projects demonstrating these skills before applying."
                )
            else:
                actionable_advice = "Ensure your resume explicitly mentions your technical tools, frameworks, and project links."

        return {
            "verdict_badge": verdict_badge,
            "readiness_summary": readiness_summary,
            "actionable_advice": actionable_advice
        }

    def assess_resume(self, resume_path_or_text: str, target_role: str, is_raw_text: bool = False) -> Dict[str, Any]:
        """
        Executes complete self-assessment workflow on an uploaded PDF resume or text string.
        """
        if target_role not in ROLE_SKILL_TAXONOMY:
            valid_roles = ", ".join(ROLE_SKILL_TAXONOMY.keys())
            raise ValueError(f"Invalid target role '{target_role}'. Must be one of: {valid_roles}")

        # Extract text
        if is_raw_text:
            text = resume_path_or_text
        else:
            text = extract_text_from_pdf(resume_path_or_text)

        # Extract and compare skills
        role_profile = ROLE_SKILL_TAXONOMY[target_role]
        core_reqs = set(role_profile["core"])
        supp_reqs = set(role_profile["supporting"])

        extracted_skills = self.extract_skills(text)
        
        matched_core = sorted(list(extracted_skills.intersection(core_reqs)))
        missing_core = sorted(list(core_reqs.difference(extracted_skills)))
        matched_supp = sorted(list(extracted_skills.intersection(supp_reqs)))
        missing_supp = sorted(list(supp_reqs.difference(extracted_skills)))

        core_pct = len(matched_core) / len(core_reqs) if core_reqs else 0.0
        supp_pct = len(matched_supp) / len(supp_reqs) if supp_reqs else 0.0
        
        proj_count, depth_score = self.evaluate_project_depth(text)

        # Composite score calculation
        composite_score = (
            (core_pct * self.w_core) +
            (supp_pct * self.w_supp) +
            (depth_score * self.w_depth)
        ) * 100.0
        
        composite_score = round(composite_score, 1)

        guidance = self.generate_guidance(composite_score, missing_core, missing_supp, proj_count, target_role)

        return {
            "target_role": target_role,
            "compatibility_score": composite_score,
            "verdict_badge": guidance["verdict_badge"],
            "readiness_summary": guidance["readiness_summary"],
            "actionable_advice": guidance["actionable_advice"],
            "core_breakdown": {
                "matched": matched_core,
                "missing": missing_core,
                "coverage_pct": round(core_pct * 100, 1),
                "count": f"{len(matched_core)} of {len(core_reqs)} matched"
            },
            "supporting_breakdown": {
                "matched": matched_supp,
                "missing": missing_supp,
                "coverage_pct": round(supp_pct * 100, 1),
                "count": f"{len(matched_supp)} of {len(supp_reqs)} matched"
            },
            "project_depth": {
                "projects_detected": proj_count,
                "depth_multiplier": round(depth_score, 2)
            }
        }

    def render_student_feedback_card(self, assessment: Dict[str, Any]) -> str:
        """Formats assessment output into a clean, encouraging UI text card."""
        score = assessment["compatibility_score"]
        role = assessment["target_role"]
        badge = assessment["verdict_badge"]
        summary = assessment["readiness_summary"]
        advice = assessment["actionable_advice"]
        core = assessment["core_breakdown"]
        supp = assessment["supporting_breakdown"]
        depth = assessment["project_depth"]

        matched_core_str = ", ".join(core["matched"]) if core["matched"] else "None detected"
        missing_core_str = ", ".join(core["missing"]) if core["missing"] else "None (100% Core Coverage!)"
        matched_supp_str = ", ".join(supp["matched"]) if supp["matched"] else "None detected"

        card = f"""
================================================================================
           PORTFOLIOMATCH AI — STUDENT SELF-ASSESSMENT REPORT
================================================================================
 TARGET ROLE: {role}
 COMPATIBILITY RATING: {score}% | {badge}
--------------------------------------------------------------------------------

[ READINESS OVERVIEW ]
{summary}

[ ACTIONABLE RECOMMENDATIONS ]
-> {advice}

--------------------------------------------------------------------------------
[ DETAILED SKILL BREAKDOWN ]

* Core Role Skills ({core['count']} | {core['coverage_pct']}%):
  - Matched: {matched_core_str}
  - Missing to Prioritize: {missing_core_str}

* Supporting Skills ({supp['count']} | {supp['coverage_pct']}%):
  - Matched: {matched_supp_str}

* Portfolio / Project Evidence:
  - Detected Projects / Repositories: {depth['projects_detected']}
  - Depth Verification Multiplier: {depth['depth_multiplier']}x

--------------------------------------------------------------------------------
Tip: Update your resume or project portfolio, then re-run this check anytime!
================================================================================
"""
        return card