import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

sys.path.append("src")

from io_utils import load_resumes_from_csv
from job_parser import parse_job_skills
from role_predictor import RolePredictor
from scoring import compute_tfidf_similarity, overlap_terms, top_job_terms
from skills import extract_skills, load_skill_patterns, skill_gap

APP_TITLE = "AI Resume Screening System"


def _load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Space+Grotesk:wght@400;600;700&display=swap');

        :root {
            --bg: #f5f1ea;
            --ink: #1e1b16;
            --muted: #6f675c;
            --accent: #0b7a75;
            --accent-2: #f1b24a;
            --card: #ffffff;
            --border: rgba(30, 27, 22, 0.08);
            --shadow: 0 18px 40px rgba(30, 27, 22, 0.12);
        }
        html, body, .stApp {
            background:
                radial-gradient(1200px 800px at 10% -10%, rgba(11, 122, 117, 0.12), transparent),
                radial-gradient(900px 600px at 120% 10%, rgba(241, 178, 74, 0.12), transparent),
                var(--bg);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }
        .block-container {
            max-width: 1100px;
            padding-top: 2.75rem;
        }
        h1, h2, h3, h4 {
            font-family: 'Fraunces', serif;
        }
        label, .stTextArea label, .stFileUploader label, .stTextInput label, .stSelectbox label {
            color: var(--ink) !important;
            font-weight: 600;
        }
        textarea, input, .stTextArea textarea {
            background: #fffdf8 !important;
            color: var(--ink) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 22px 26px;
            border-radius: 20px;
            background:
                linear-gradient(140deg, rgba(11,122,117,0.08), rgba(241,178,74,0.08)),
                repeating-linear-gradient(45deg, rgba(30,27,22,0.03), rgba(30,27,22,0.03) 2px, transparent 2px, transparent 8px);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 22px;
            animation: fadeUp 0.6s ease-out;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0;
        }
        .hero-sub {
            color: var(--muted);
            margin-top: 6px;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #fff;
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.9rem;
            color: var(--muted);
        }
        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
            animation: fadeUp 0.6s ease-out;
        }
        .panel-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .stat-card {
            background: #fff;
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }
        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent);
        }
        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(11, 122, 117, 0.12);
            color: var(--accent);
            font-size: 0.85rem;
            margin: 4px 6px 0 0;
            border: 1px solid rgba(11, 122, 117, 0.25);
        }
        .pill.missing {
            background: rgba(231, 76, 60, 0.12);
            color: #b13e34;
            border: 1px solid rgba(231, 76, 60, 0.3);
        }
        .upload {
            border: 1px dashed rgba(30,27,22,0.2);
            border-radius: 14px;
            padding: 18px;
            background: #fffdf8;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        #MainMenu, footer {visibility: hidden;}
        @keyframes fadeUp {
            from {opacity: 0; transform: translateY(8px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_uploaded_file(uploaded) -> str:
    if not uploaded:
        return ""
    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            st.error("PyPDF2 is required to read PDF files. Install it with pip install PyPDF2")
            raise exc
        reader = PdfReader(uploaded)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    return uploaded.read().decode("utf-8", errors="ignore")


@st.cache_resource
def _load_role_predictor() -> RolePredictor:
    resumes_path = Path("Resume.csv") if Path("Resume.csv").exists() else Path("data/resumes.csv")
    df = load_resumes_from_csv(resumes_path)
    if "category" not in df.columns:
        return RolePredictor()
    predictor = RolePredictor()
    predictor.fit(df["text"].tolist(), df["category"].tolist())
    return predictor


@st.cache_resource
def _load_skill_patterns():
    return load_skill_patterns("data/skills.json")


def _score_resume(job_text: str, resume_text: str) -> Tuple[float, float, float, List[str], List[str], List[str], List[str], List[str]]:
    patterns = _load_skill_patterns()
    required_skills, nice_skills = parse_job_skills(job_text, patterns)

    similarities, vectorizer, tfidf = compute_tfidf_similarity(job_text, [resume_text])
    similarity_score = float(similarities[0])

    found_skills, _ = extract_skills(resume_text, patterns)
    matched_required, missing_required = skill_gap(required_skills, found_skills)
    matched_nice, missing_nice = skill_gap(nice_skills, found_skills)

    required_score = len(matched_required) / len(required_skills) if required_skills else 0.0
    nice_score = len(matched_nice) / len(nice_skills) if nice_skills else 0.0

    skill_score = required_score if required_skills else nice_score
    if required_skills and nice_skills:
        skill_score = 0.7 * required_score + 0.3 * nice_score

    total_score = (0.6 * similarity_score) + (0.4 * skill_score)

    job_terms = top_job_terms(vectorizer, tfidf, top_n=8)
    resume_vec = tfidf[1:2]
    overlap = overlap_terms(resume_vec, job_terms, vectorizer)

    return (
        total_score,
        similarity_score,
        skill_score,
        matched_required,
        missing_required,
        matched_nice,
        missing_nice,
        overlap,
    )


def _render_pills(items: List[str], missing: bool = False) -> None:
    if not items:
        st.markdown("<span class='subtle'>None</span>", unsafe_allow_html=True)
        return
    html = "".join(
        f"<span class='pill{' missing' if missing else ''}'>{item}</span>" for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(page_title=APP_TITLE, layout="wide")
_load_css()

st.markdown(
    """
    <div class='hero'>
        <div>
            <div class='hero-title'>Candidate Matchroom</div>
            <div class='hero-sub'>A focused, human-friendly view of fit, gaps, and role prediction.</div>
        </div>
        <div class='chip'>Resume Intelligence • v1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel-title'>Job Brief</div>", unsafe_allow_html=True)
    job_text = st.text_area(
        label="Job Description",
        value=Path("data/job_description.txt").read_text(encoding="utf-8")
        if Path("data/job_description.txt").exists()
        else "",
        height=160,
    )

with col2:
    st.markdown("<div class='panel-title'>Resume File</div>", unsafe_allow_html=True)
    st.markdown("<div class='upload'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a resume (.pdf or .txt)", type=["pdf", "txt"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded and job_text.strip():
    resume_text = _read_uploaded_file(uploaded)

    if resume_text.strip():
        predictor = _load_role_predictor()
        predicted_role, role_scores = predictor.predict(resume_text)

        (
            total_score,
            similarity_score,
            skill_score,
            matched_required,
            missing_required,
            matched_nice,
            missing_nice,
            overlap,
        ) = _score_resume(job_text, resume_text)

        st.markdown("<div class='panel-title'>Results</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Predicted Role</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-value'>{predicted_role or 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Match Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-value'>{total_score * 100:.2f}%</div>", unsafe_allow_html=True)
            st.progress(min(max(total_score, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Required Skills</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Matched</div>", unsafe_allow_html=True)
        _render_pills(matched_required)
        st.markdown("<div class='subtle' style='margin-top:8px;'>Missing</div>", unsafe_allow_html=True)
        _render_pills(missing_required, missing=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Nice-to-Haves</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Matched</div>", unsafe_allow_html=True)
        _render_pills(matched_nice)
        st.markdown("<div class='subtle' style='margin-top:8px;'>Missing</div>", unsafe_allow_html=True)
        _render_pills(missing_nice, missing=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Job Term Overlap</div>", unsafe_allow_html=True)
        _render_pills(overlap)
        st.markdown("</div>", unsafe_allow_html=True)

        if role_scores:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Top Role Scores</div>", unsafe_allow_html=True)
            top_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for role, score in top_roles:
                st.markdown(f"<div class='subtle'>{role}: {score * 100:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Uploaded file is empty or could not be read.")
else:
    st.info("Upload a resume and add a job description to see results.")
