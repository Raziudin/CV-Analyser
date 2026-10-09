import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from crewai import Agent, Task, Crew, Process, LLM

# Local development only. On Streamlit Cloud this file doesn't exist and the
# Secrets box values are available as environment variables / st.secrets.
load_dotenv(Path(__file__).parent / "keys" / "keys.env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cv_analyser")

st.set_page_config(page_title="CV Analyser", page_icon="📄", layout="wide")
st.title("📄 CV Analyser (CrewAI agents)")


# ---------- Secrets helper ----------
def get_secret(name):
    """Read from env first, then st.secrets. Never raises."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# ---------- Providers ----------
PROVIDERS = {
    "Groq": ("openai/openai/gpt-oss-120b", "GROQ_API_KEY"),
    "xAI Grok": ("openai/grok-4", "XAI_API_KEY"),
    "Google Gemini": ("gemini/gemini-2.0-flash", "GEMINI_API_KEY"),
    "OpenAI": ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    "Anthropic": ("anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
}
# Ollama only works when the app runs on your own machine.
if os.getenv("ENABLE_OLLAMA") == "1":
    PROVIDERS["Ollama (local)"] = ("ollama/llama3.1", None)

BASE_URLS = {
    "Groq": "https://api.groq.com/openai/v1",
    "xAI Grok": "https://api.x.ai/v1",
}

APP_PASSWORD = get_secret("APP_PASSWORD")  # optional access code

# ---------- Sidebar ----------
with st.sidebar:
    provider = st.selectbox("Provider", list(PROVIDERS))
    default_model, key_env = PROVIDERS[provider]
    model = st.text_input("Model", default_model)

    # Never prefill: a prefilled value is sent to the visitor's browser.
    user_key = ""
    if key_env:
        user_key = st.text_input(
            "Your API key (optional)",
            type="password",
            value="",
            help="Leave blank to use the server's key (access code may be required).",
        )

    access_code = ""
    if APP_PASSWORD and not user_key:
        access_code = st.text_input("Access code", type="password", value="")


def get_key():
    """Visitor's own key wins; otherwise the server-side key. Kept local, never
    written to os.environ (the process is shared by all visitors)."""
    if not key_env:
        return None
    return user_key or get_secret(key_env)


# ---------- Helpers ----------
def read_cv(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


def build_crew(cv, target, mode, key):
    kw = {"model": model, "temperature": 0.2}
    if key:
        kw["api_key"] = key
    if provider == "Ollama (local)":
        kw["base_url"] = "http://localhost:11434"
    if provider in BASE_URLS:  # OpenAI-compatible endpoints
        kw["base_url"] = BASE_URLS[provider]
    llm = LLM(**kw)

    parser = Agent(
        role="CV Parser",
        goal="Extract structured data (education, skills, projects, experience, achievements) from the CV.",
        backstory="Precise information extractor. Never invents facts.",
        llm=llm, allow_delegation=False)
    matcher = Agent(
        role="Fit Evaluator",
        goal=f"Score how well the CV fits the target ({mode}) and find gaps and missing keywords.",
        backstory="Experienced recruiter / admissions reviewer and ATS expert.",
        llm=llm, allow_delegation=False)
    coach = Agent(
        role="CV Coach",
        goal="Give concrete, actionable improvements and rewrite weak bullet points.",
        backstory="Career coach: direct, specific, evidence-based, never fabricates experience.",
        llm=llm, allow_delegation=False)

    t1 = Task(
        description=f"Parse this CV into structured sections.\n\nCV:\n{cv}",
        expected_output="Structured summary of the CV.", agent=parser)
    t2 = Task(
        description=f"Evaluate fit against this target {mode}:\n\n{target}\n\n"
        "Give: overall score /100, score breakdown (skills, experience, education, keywords), "
        "strengths, gaps, missing keywords.",
        expected_output="Scored evaluation with gaps and keywords.",
        agent=matcher, context=[t1])
    t3 = Task(
        description="Produce: 1) top 5 prioritized fixes, 2) 5 rewritten bullet points "
        "(only using facts in the CV), 3) formatting/ATS issues, 4) a 3-line tailored summary. "
        "Output in Markdown.",
        expected_output="Markdown improvement report.",
        agent=coach, context=[t1, t2])
    return Crew(agents=[parser, matcher, coach], tasks=[t1, t2, t3],
                process=Process.sequential)


# ---------- UI ----------
col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Upload CV (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
with col2:
    mode = st.selectbox("Analyse against",
                        ["Job description", "Master's/PhD program", "General review"])
    target = st.text_area("Paste job / program description",
                          height=180, disabled=(mode == "General review"))

st.caption("Your CV is processed on the server and sent to the selected LLM provider.")

if st.button("Analyse", type="primary"):
    if not file:
        st.error("Upload a CV first.")
        st.stop()

    # Access gate applies only when using the server's key
    if APP_PASSWORD and not user_key and access_code != APP_PASSWORD:
        st.error("Enter the access code in the sidebar, or use your own API key.")
        st.stop()

    key = get_key()
    if key_env and not key:
        st.error("No API key available. Paste your own key in the sidebar.")
        st.stop()

    cv = read_cv(file)
    if len(cv.strip()) < 100:
        st.error("Couldn't extract text (scanned PDF?). Try DOCX or a text-based PDF.")
        st.stop()

    if mode == "General review":
        target = "General strong technical CV for entry-level engineering/AI roles."

    with st.spinner("Agents working..."):
        try:
            result = str(build_crew(cv, target, mode, key).kickoff())
            st.markdown(result)
            st.download_button("Download report", result, "cv_report.md")
        except Exception:
            log.exception("Crew run failed")  # details stay in server logs
            st.error("Analysis failed. Check the model name and API key, then try again.")
