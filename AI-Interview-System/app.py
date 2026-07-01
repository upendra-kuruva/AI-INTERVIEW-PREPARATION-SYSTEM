"""
AI Interview Preparation System - Streamlit app.

Flow:
  1. Enter Gemini API key (sidebar)
  2. Upload resume -> Resume Agent extracts skills/strengths/weak areas
  3. Generate personalized questions (RAG + HR/Python/AI agents)
  4. Answer questions one by one -> LangGraph routes to specialist + feedback agent
  5. Final report with scores, weak areas, recommended topics, downloadable PDF
"""
import os
import io
import tempfile

import streamlit as st
from pypdf import PdfReader
import docx

from coordinator import InterviewCoordinator

st.set_page_config(page_title="AI Interview Preparation System", page_icon="🎤", layout="centered")


# ---------- helpers ----------

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    return data.decode("utf-8", errors="ignore")


def reset_session():
    for key in ["coordinator", "phase", "resume_text", "last_feedback"]:
        st.session_state.pop(key, None)


# ---------- sidebar ----------

st.sidebar.title("Setup")
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get a free key at https://aistudio.google.com/apikey",
)
if st.sidebar.button("Reset session"):
    reset_session()
    st.rerun()

st.sidebar.caption(
    "This app uses Google Gemini for all agents, LangGraph for multi-agent routing, "
    "and a local FAISS index (RAG) over a seed interview question bank."
)

st.title("AI Interview Preparation System")

if "phase" not in st.session_state:
    st.session_state.phase = "upload"

# ---------- phase: upload & analyze resume ----------

if st.session_state.phase == "upload":
    st.subheader("1. Upload your resume")
    uploaded = st.file_uploader("Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    num_per_domain = st.slider("Questions per domain (HR / Python / AI)", 2, 6, 3)

    if uploaded and st.button("Analyze Resume & Start Interview", type="primary"):
        if not api_key_input:
            st.error("Please enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Analyzing resume and generating personalized questions..."):
                try:
                    resume_text = extract_text_from_upload(uploaded)
                    coordinator = InterviewCoordinator(api_key_input)
                    resume_analysis = coordinator.start(resume_text, num_per_domain=num_per_domain)
                    st.session_state.coordinator = coordinator
                    st.session_state.resume_analysis = resume_analysis
                    st.session_state.phase = "interview"
                    st.session_state.last_feedback = None
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Something went wrong: {e}")

# ---------- phase: interview loop ----------

elif st.session_state.phase == "interview":
    coordinator: InterviewCoordinator = st.session_state.coordinator

    with st.expander(" Resume Analysis", expanded=False):
        ra = st.session_state.resume_analysis
        st.write("**Skills:**", ", ".join(ra.get("skills", [])) or "—")
        st.write("**Strengths:**", ", ".join(ra.get("strengths", [])) or "—")
        st.write("**Areas to watch:**", ", ".join(ra.get("weak_areas", [])) or "—")

    if coordinator.is_finished():
        st.session_state.phase = "report"
        st.rerun()

    total = coordinator.total_questions()
    idx = coordinator.index
    st.progress(idx / total if total else 0, text=f"Question {idx + 1} of {total}")

    q = coordinator.current_question()
    badge = f"`{q['domain'].upper()}`  ·  difficulty: `{q.get('difficulty','medium')}`"
    st.markdown(badge)
    st.markdown(f"### {q['question']}")

    if st.session_state.get("last_feedback") and st.session_state.last_feedback["index"] == idx:
        fb = st.session_state.last_feedback["data"]
        st.success(fb["feedback"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correctness", f"{fb['correctness']}/10")
        c2.metric("Clarity", f"{fb['clarity']}/10")
        c3.metric("Confidence", f"{fb['confidence']}/10")
        c4.metric("Tech Depth", f"{fb['technical_depth']}/10")
        if fb.get("missing_points"):
            st.caption("Consider also mentioning: " + ", ".join(fb["missing_points"]))
        if st.button("Next Question ➡️", type="primary"):
            coordinator.advance()
            st.session_state.last_feedback = None
            st.rerun()
    else:
        answer = st.text_area("Your answer", key=f"answer_{idx}", height=150)
        if st.button("Submit Answer", type="primary"):
            if not answer.strip():
                st.warning("Please write an answer before submitting.")
            else:
                with st.spinner("Evaluating your answer..."):
                    try:
                        feedback = coordinator.submit_answer(answer)
                        st.session_state.last_feedback = {"index": idx, "data": feedback}
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Evaluation failed: {e}")

# ---------- phase: final report ----------

elif st.session_state.phase == "report":
    coordinator: InterviewCoordinator = st.session_state.coordinator
    report = coordinator.finalize()
    summary = report["summary"]

    st.subheader("📊 Final Report")
    st.metric("Overall Score", f"{summary['overall_score']} / 10")

    st.write("**Scores by domain**")
    if summary["domain_scores"]:
        st.bar_chart(summary["domain_scores"])

    st.write("**Weak areas:**", ", ".join(summary["weak_areas"]) or "None — solid all-around!")
    st.write("**Recommended topics to review:**")
    for topic in summary["recommended_topics"]:
        st.markdown(f"- {topic}")

    with st.expander("Full Q&A transcript"):
        for i, turn in enumerate(report["transcript"], 1):
            st.markdown(f"**{i}. [{turn['domain'].upper()}] {turn['question']}**")
            st.write(f"Answer: {turn['answer']}")
            st.write(f"Score: {turn['overall_score']}/10 — {turn['feedback']}")
            st.divider()

    if st.button("Generate PDF Report"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            coordinator.export_pdf(tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
        st.download_button(
            "⬇️ Download PDF Report", data=pdf_bytes,
            file_name="interview_report.pdf", mime="application/pdf",
        )

    if st.button("Start a new interview"):
        reset_session()
        st.rerun()
