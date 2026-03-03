import streamlit as st
from model_utils import compute_match_score
from genai_utils import build_prompt, get_explanation
from pdf_utils import extract_text_from_pdf

st.set_page_config(
    page_title="GenAI Resume Matcher",
    layout="wide"
)

st.title("🧠 GenAI Resume ↔ Job Matching System")
st.caption("Hybrid ML + Semantic Search + GenAI Reasoning")

# --- Resume Input ---
st.subheader("📄 Resume Input")

resume_mode = st.radio(
    "Choose resume input method:",
    ["Paste Text", "Upload PDF"]
)

resume_text = ""

if resume_mode == "Paste Text":
    resume_text = st.text_area("Paste Resume Text", height=250)

else:
    uploaded_pdf = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_pdf)

        with st.expander("📄 Extracted Resume Text (Preview)"):
            st.write(resume_text[:2000])

# --- Job Description ---
st.subheader("🧾 Job Description")
job_text = st.text_area("Paste Job Description", height=250)

# --- Evaluation ---
if st.button("Evaluate Match"):
    if not resume_text or not job_text:
        st.warning("Please provide both resume and job description.")
    else:
        with st.spinner("Scoring resume with ML model..."):
            score, features = compute_match_score(resume_text, job_text)

        st.metric("Match Score (0–10)", f"{score:.2f}")

        with st.expander("🔍 Feature Breakdown"):
            st.json(features)

        with st.spinner("Generating recruiter explanation..."):
            prompt = build_prompt(resume_text, job_text, score)
            explanation = get_explanation(prompt)

        st.subheader("🧠 Recruiter Explanation")
        st.write(explanation)
