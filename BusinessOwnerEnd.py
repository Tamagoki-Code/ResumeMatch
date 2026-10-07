"""
PortfolioMatch AI — ML Model Baseline
Component: Rule-Based Skill Extraction, Resume Scoring, and Applicant Ranking Engine
"""

import json
import math
import re
from typing import Any, Dict, List, Set, Tuple


# ==========================================
# 1. SKILL TAXONOMY & ROLE DEFINITIONS
# Grounded in O*NET & roadmap.sh standards
# ==========================================

ROLE_SKILL_PROFILES: Dict[str, Dict[str, List[str]]] = {
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

SKILL_CANONICAL_MAP: Dict[str, str] = {
    # Web & Languages
    "html": "HTML", "html5": "HTML",
    "css": "CSS", "css3": "CSS",
    "javascript": "JavaScript", "js": "JavaScript", "ecmascript": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "python": "Python", "python3": "Python",
    "java": "Java", "dart": "Dart",
    "kotlin": "Kotlin", "swift": "Swift",
    
    # Frontend Frameworks & Libraries
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
    
    # Backend & DB
    "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js",
    "express": "Express.js", "express.js": "Express.js", "expressjs": "Express.js",
    "django": "Django", "fastapi": "FastAPI",
    "sql": "SQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mysql": "MySQL", "mongodb": "MongoDB", "mongo": "MongoDB",
    "redis": "Redis", "rest api": "REST API", "restful api": "REST API", "rest apis": "REST API", "rest": "REST API",
    
    # Data & ML
    "pandas": "Pandas", "numpy": "NumPy",
    "scikit-learn": "Scikit-Learn", "sklearn": "Scikit-Learn",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "data analysis": "Data Analysis", "data analytics": "Data Analysis",
    "data visualization": "Data Visualization", "matplotlib": "Matplotlib", "seaborn": "Data Visualization",
    "jupyter": "Jupyter", "jupyter notebook": "Jupyter",
    
    # UI/UX
    "figma": "Figma", "adobe xd": "Adobe XD", "xd": "Adobe XD",
    "wireframing": "Wireframing", "wireframes": "Wireframing", "wireframe": "Wireframing",
    "prototyping": "Prototyping", "prototype": "Prototyping", "prototypes": "Prototyping",
    "user research": "User Research", "usability testing": "Usability Testing",
    "design systems": "Design Systems", "design system": "Design Systems",
    "information architecture": "Information Architecture", "ia": "Information Architecture",
    "photoshop": "Photoshop", "interaction design": "Interaction Design",
    
    # Cloud & Tools
    "git": "Git", "github": "Git", "version control": "Git",
    "docker": "Docker", "aws": "AWS", "amazon web services": "AWS",
    "firebase": "Firebase"
}


# ==========================================
# 2. EXTRACTION & SCORING PIPELINE
# ==========================================

class PortfolioMatchScorer:
    def __init__(self, core_weight: float = 0.60, supporting_weight: float = 0.25, depth_weight: float = 0.15):
        self.w_core = core_weight
        self.w_supp = supporting_weight
        self.w_depth = depth_weight

    def extract_skills(self, text: str) -> Set[str]:
        """Extracts canonical skills from resume text using regex word-boundary matching."""
        found_skills: Set[str] = set()
        clean_text = text.lower()
        
        # Sort keys by length descending to match multi-word phrases first
        sorted_patterns = sorted(SKILL_CANONICAL_MAP.keys(), key=lambda x: len(x), reverse=True)
        
        for pattern in sorted_patterns:
            escaped_pattern = re.escape(pattern)
            # Match with boundary check (handles punctuation-based names like node.js and c++)
            regex = rf"(?:\b|\A){escaped_pattern}(?:\b|\Z)"
            if re.search(regex, clean_text):
                canonical_name = SKILL_CANONICAL_MAP[pattern]
                found_skills.add(canonical_name)
                
        return found_skills

    def extract_project_depth(self, text: str) -> Tuple[int, float]:
        """
        Heuristic project depth proxy: counts project entries and calculates 
        normalized project depth signal (0.0 to 1.0).
        """
        lines = text.split("\n")
        project_count = 0
        in_project_section = False
        
        for line in lines:
            lower = line.strip().lower()
            if "project" in lower and len(lower) < 30:
                in_project_section = True
                continue
            if in_project_section and any(sec in lower for sec in ["education", "experience", "certifications", "skills"]):
                in_project_section = False
                
            if in_project_section and (line.strip().startswith("-") or line.strip().startswith("•") or "http" in lower or "github" in lower):
                project_count += 1

        # Fallback keyword scanning for project count
        if project_count == 0:
            project_count = len(re.findall(r"(?:project\s*\d+|github\.com\/|built\s+a|developed\s+a)", text, re.IGNORECASE))

        # Normalize: 0 projects = 0.2, 1 = 0.5, 2 = 0.75, 3+ = 1.0
        if project_count == 0:
            depth_score = 0.25
        elif project_count == 1:
            depth_score = 0.55
        elif project_count == 2:
            depth_score = 0.80
        else:
            depth_score = 1.00
            
        return project_count, depth_score

    def score_applicant(self, resume_id: str, applicant_name: str, resume_text: str, target_role: str) -> Dict[str, Any]:
        """Calculates role-specific match metrics and composite percentage score."""
        if target_role not in ROLE_SKILL_PROFILES:
            raise ValueError(f"Unknown target role: {target_role}")

        profile = ROLE_SKILL_PROFILES[target_role]
        core_skills_required = set(profile["core"])
        supp_skills_required = set(profile["supporting"])

        extracted_skills = self.extract_skills(resume_text)
        
        matched_core = extracted_skills.intersection(core_skills_required)
        missing_core = core_skills_required.difference(matched_core)
        matched_supp = extracted_skills.intersection(supp_skills_required)
        
        core_match_pct = len(matched_core) / len(core_skills_required) if core_skills_required else 0.0
        supp_match_pct = len(matched_supp) / len(supp_skills_required) if supp_skills_required else 0.0
        
        proj_count, depth_score = self.extract_project_depth(resume_text)

        composite_score = (
            (core_match_pct * self.w_core) +
            (supp_match_pct * self.w_supp) +
            (depth_score * self.w_depth)
        ) * 100.0

        # Tier assignment
        if composite_score >= 75.0:
            assigned_tier = "Strong Match"
        elif composite_score >= 45.0:
            assigned_tier = "Medium Match"
        else:
            assigned_tier = "Weak Match"

        return {
            "resume_id": resume_id,
            "applicant_name": applicant_name,
            "target_role": target_role,
            "score": round(composite_score, 1),
            "tier": assigned_tier,
            "core_match_pct": round(core_match_pct * 100, 1),
            "supp_match_pct": round(supp_match_pct * 100, 1),
            "matched_core": sorted(list(matched_core)),
            "missing_core": sorted(list(missing_core)),
            "matched_supporting": sorted(list(matched_supp)),
            "project_count": proj_count,
            "depth_score": round(depth_score, 2)
        }

    def rank_applicants(self, applicants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts applicants descending by score, breaking ties by Core Match %."""
        sorted_applicants = sorted(
            applicants,
            key=lambda x: (x["score"], x["core_match_pct"]),
            reverse=True
        )
        for rank, applicant in enumerate(sorted_applicants, start=1):
            applicant["rank"] = rank
        return sorted_applicants


# ==========================================
# 3. EVALUATION METRICS (NDCG & Spearman)
# ==========================================

def compute_dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def compute_ndcg(actual_rels: List[int], ideal_rels: List[int]) -> float:
    dcg = compute_dcg(actual_rels)
    idcg = compute_dcg(sorted(ideal_rels, reverse=True))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0

def compute_spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Computes Spearman's rank correlation coefficient."""
    n = len(x)
    if n <= 1:
        return 1.0
    
    def get_ranks(seq: List[float]) -> List[float]:
        sorted_seq = sorted(enumerate(seq), key=lambda item: item[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_seq[j][1] == sorted_seq[j + 1][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_seq[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_x = get_ranks(x)
    rank_y = get_ranks(y)
    
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    
    num = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    den = math.sqrt(sum((rx - mean_x) ** 2 for rx in rank_x) * sum((ry - mean_y) ** 2 for ry in rank_y))
    
    return round(num / den, 4) if den != 0 else 0.0


# ==========================================
# 4. BENCHMARK SIMULATION DATASET (20 RESUMES)
# ==========================================

BENCHMARK_RESUMES = [
    # --- Web Developer (4 candidates) ---
    {
        "id": "RES_WD_01", "name": "Alden Rivera", "target_role": "Web Developer", "ground_truth": "Strong",
        "text": "Alden Rivera - Frontend Developer. Skills: HTML5, CSS3, JavaScript (ES6+), React.js, Responsive Design, Git, TypeScript, Tailwind CSS, REST APIs. Projects: 1. MSME E-Commerce Platform built with React, Tailwind, and REST API. 2. Portfolio Showcase using Responsive Design and Git. 3. Task Management App using TypeScript and Node.js."
    },
    {
        "id": "RES_WD_02", "name": "Bea Alonzo", "target_role": "Web Developer", "ground_truth": "Medium",
        "text": "Bea Alonzo - Web Designer & Developer. Skills: HTML, CSS, JavaScript, Bootstrap, Responsive Design, Git. Experience: Created 2 static websites for local bakeries. Used HTML/CSS and basic JS DOM manipulation. Version control with Git."
    },
    {
        "id": "RES_WD_03", "name": "Carl Dizon", "target_role": "Web Developer", "ground_truth": "Medium",
        "text": "Carl Dizon - Junior Frontend. Skills: HTML5, CSS, ReactJS, Git, Webpack. Basic understanding of Responsive Design. Projects: React portfolio app deployed on GitHub Pages."
    },
    {
        "id": "RES_WD_04", "name": "Danilo Cruz", "target_role": "Web Developer", "ground_truth": "Weak",
        "text": "Danilo Cruz - IT Graduate. Skills: Microsoft Office, Hardware Troubleshooting, Basic HTML, Customer Service. Familiar with Windows OS and networking fundamentals."
    },

    # --- Mobile Developer (3 candidates) ---
    {
        "id": "RES_MD_01", "name": "Ella Pascual", "target_role": "Mobile Developer", "ground_truth": "Strong",
        "text": "Ella Pascual - Mobile Application Engineer. Proficient in Kotlin, Swift, Flutter, React Native, Android SDK, and iOS SDK. Supporting: Firebase, REST API, Dart, Mobile UI. Projects: 1. Food Delivery App (Flutter, Firebase, REST API). 2. Fitness Tracker (Kotlin, Android SDK). 3. iOS Habit Tracker (Swift, iOS SDK)."
    },
    {
        "id": "RES_MD_02", "name": "Francis Tolentino", "target_role": "Mobile Developer", "ground_truth": "Medium",
        "text": "Francis Tolentino - Android Developer. Skills: Kotlin, Java, Android SDK, Git, Firebase. Developed 1 academic capstone project for inventory tracking on Android devices."
    },
    {
        "id": "RES_MD_03", "name": "Giselle Tan", "target_role": "Mobile Developer", "ground_truth": "Weak",
        "text": "Giselle Tan - Junior Programmer. Skills: Python, Basic Java, HTML, CSS. Interested in learning mobile app development and cross-platform tools."
    },

    # --- UI/UX Designer (3 candidates) ---
    {
        "id": "RES_UI_01", "name": "Hannah Sotto", "target_role": "UI/UX Designer", "ground_truth": "Strong",
        "text": "Hannah Sotto - Product & UI/UX Designer. Expertise: Figma, Adobe XD, Wireframing, Prototyping, User Research, Usability Testing. Supporting: Design Systems, Information Architecture, Interaction Design, HTML/CSS. Projects: Redesign of sari-sari store POS interface in Figma; usability tested on 15 business owners."
    },
    {
        "id": "RES_UI_02", "name": "Ian De Leon", "target_role": "UI/UX Designer", "ground_truth": "Medium",
        "text": "Ian De Leon - Visual Designer. Skills: Figma, Adobe XD, Wireframing, Photoshop, Graphic Design. Created mockup designs for 2 client websites."
    },
    {
        "id": "RES_UI_03", "name": "Joy Mercado", "target_role": "UI/UX Designer", "ground_truth": "Weak",
        "text": "Joy Mercado - Digital Marketing Specialist. Skills: Photoshop, Canva, Social Media Management, Basic HTML. Created marketing banners and social media posters."
    },

    # --- Backend Developer (3 candidates) ---
    {
        "id": "RES_BE_01", "name": "Kevin Ramos", "target_role": "Backend Developer", "ground_truth": "Strong",
        "text": "Kevin Ramos - Backend Engineer. Core Competencies: Node.js, Python, SQL, REST API, PostgreSQL, MongoDB. Supporting: Docker, Redis, Express.js, Django, AWS, Git. Projects: 1. Microservices E-Commerce API (Node.js, Express, MongoDB, Docker). 2. High-throughput data caching service (Python, Redis, PostgreSQL)."
    },
    {
        "id": "RES_BE_02", "name": "Lara Santos", "target_role": "Backend Developer", "ground_truth": "Medium",
        "text": "Lara Santos - Software Developer. Skills: Python, Django, SQL, PostgreSQL, Git. Built a university student portal backend using Django and PostgreSQL with RESTful endpoints."
    },
    {
        "id": "RES_BE_03", "name": "Marco Perez", "target_role": "Backend Developer", "ground_truth": "Weak",
        "text": "Marco Perez - Web Enthusiast. Skills: HTML, CSS, JavaScript, WordPress, PHP. Experienced in setting up shared hosting and WordPress blogs."
    },

    # --- Full Stack Developer (4 candidates) ---
    {
        "id": "RES_FS_01", "name": "Nathan Reyes", "target_role": "Full Stack Developer", "ground_truth": "Strong",
        "text": "Nathan Reyes - Full Stack Developer. Core: JavaScript, React, Node.js, SQL, REST API, Git. Supporting: TypeScript, Express.js, PostgreSQL, MongoDB, Docker, AWS, HTML, CSS. Projects: 1. PortfolioMatch Web App (React, Express, Node.js, PostgreSQL). 2. Inventory System (TypeScript, MongoDB, AWS). 3. Realtime Chat Platform (React, Socket.io, Docker)."
    },
    {
        "id": "RES_FS_02", "name": "Olivia Cortez", "target_role": "Full Stack Developer", "ground_truth": "Medium",
        "text": "Olivia Cortez - MERN Stack Developer. Skills: JavaScript, React, Node.js, Express, MongoDB, HTML, CSS, Git. Projects: Developed an online food ordering web app using React, Node.js, and MongoDB."
    },
    {
        "id": "RES_FS_03", "name": "Paulo Garcia", "target_role": "Full Stack Developer", "ground_truth": "Medium",
        "text": "Paulo Garcia - Web Application Developer. Skills: JavaScript, HTML, CSS, Node.js, SQL, REST APIs. Built 1 class project database application."
    },
    {
        "id": "RES_FS_04", "name": "Queen Hernandez", "target_role": "Full Stack Developer", "ground_truth": "Weak",
        "text": "Queen Hernandez - IT Support Specialist. Skills: Technical Support, Network Configuration, Basic SQL, Excel. Aspiring full stack developer."
    },

    # --- Data/ML Developer (3 candidates) ---
    {
        "id": "RES_ML_01", "name": "Rafael Aquino", "target_role": "Data/ML Developer", "ground_truth": "Strong",
        "text": "Rafael Aquino - Machine Learning Engineer. Skills: Python, SQL, Pandas, Scikit-Learn, Machine Learning, Data Analysis, NumPy, TensorFlow, PyTorch, Matplotlib, Jupyter, Git. Projects: 1. Predictive Churn Model with Scikit-Learn and Pandas. 2. Deep Learning Image Classifier using TensorFlow. 3. Customer Segmentation Pipeline with SQL and Data Analysis."
    },
    {
        "id": "RES_ML_02", "name": "Sophia Lim", "target_role": "Data/ML Developer", "ground_truth": "Medium",
        "text": "Sophia Lim - Data Analyst. Skills: Python, SQL, Pandas, Data Analysis, Matplotlib, Data Visualization, Jupyter. Analyzed MSME retail transaction logs and built descriptive dashboards."
    },
    {
        "id": "RES_ML_03", "name": "Tristan Go", "target_role": "Data/ML Developer", "ground_truth": "Weak",
        "text": "Tristan Go - Business Analyst. Skills: Excel, PowerBI, Basic SQL, Presentations. Experience creating sales reports and summary tables."
    }
]


# ==========================================
# 5. EXECUTION & RESULTS GENERATION
# ==========================================

def run_baseline_pipeline():
    scorer = PortfolioMatchScorer()
    scored_pool: List[Dict[str, Any]] = []

    for resume in BENCHMARK_RESUMES:
        result = scorer.score_applicant(
            resume_id=resume["id"],
            applicant_name=resume["name"],
            resume_text=resume["text"],
            target_role=resume["target_role"]
        )
        result["ground_truth"] = resume["ground_truth"]
        scored_pool.append(result)

    # Group and Rank by Role
    roles = list(ROLE_SKILL_PROFILES.keys())
    ranked_by_role: Dict[str, List[Dict[str, Any]]] = {}
    
    truth_map = {"Strong": 3, "Medium": 2, "Weak": 1}
    
    print("=" * 90)
    print("PORTFOLIOMATCH AI: ML MODEL BASELINE EXECUTION SUMMARY")
    print("=" * 90)
    
    all_scores = []
    all_truth_vals = []
    
    for role in roles:
        role_candidates = [c for c in scored_pool if c["target_role"] == role]
        ranked = scorer.rank_applicants(role_candidates)
        ranked_by_role[role] = ranked
        
        # Calculate NDCG & Spearman per role
        actual_rels = [truth_map[c["ground_truth"]] for c in ranked]
        ideal_rels = sorted(actual_rels, reverse=True)
        ndcg_val = compute_ndcg(actual_rels, ideal_rels)
        
        role_scores = [c["score"] for c in ranked]
        role_truths = [truth_map[c["ground_truth"]] for c in ranked]
        spearman_val = compute_spearman_rank_correlation(role_scores, role_truths)
        
        all_scores.extend(role_scores)
        all_truth_vals.extend(role_truths)

        print(f"\nROLE: {role.upper()} (NDCG: {ndcg_val:.4f} | Spearman rho: {spearman_val:.4f})")
        print("-" * 90)
        print(f"{'Rank':<5}{'ID':<12}{'Name':<18}{'Truth':<10}{'Score':<10}{'Tier':<16}{'Matched Core':<20}")
        print("-" * 90)
        for cand in ranked:
            core_str = ", ".join(cand["matched_core"][:3]) + ("..." if len(cand["matched_core"]) > 3 else "")
            print(f"{cand['rank']:<5}{cand['resume_id']:<12}{cand['applicant_name']:<18}{cand['ground_truth']:<10}{cand['score']:<10.1f}{cand['tier']:<16}{core_str:<20}")

    # Global Spearman
    overall_spearman = compute_spearman_rank_correlation(all_scores, all_truth_vals)
    print("\n" + "=" * 90)
    print(f"OVERALL BENCHMARK SPEARMAN RANK CORRELATION (N=20): {overall_spearman:.4f}")
    print("=" * 90)

    return ranked_by_role


if __name__ == "__main__":
    run_baseline_pipeline()