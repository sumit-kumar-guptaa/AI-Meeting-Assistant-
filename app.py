"""
Streamlit UI for the AI Video/Meeting Assistant.

Assumes your existing pipeline code (the `run_pipeline` function you shared,
plus its imports: process_input, transcribe_all, summarize, generate_title,
extract_action_items, extract_key_decisions, extract_questions,
build_rag_chain, ask_question) lives in a file called `main.py` in the same
folder as this script. If it's named something else, just change the
import line below (`from main import run_pipeline, ...`).

Run with:
    streamlit run streamlit_app.py
"""

import time
import tempfile
import os
from datetime import datetime

import streamlit as st

# ---- Import your existing pipeline -----------------------------------
# Change "main" to whatever you name the file containing run_pipeline().
from main import run_pipeline
from core.rag_engine import ask_question


# =========================================================================
# PAGE CONFIG + GLOBAL STYLE
# =========================================================================
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background: #0f1117; }

    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #9ca3af; font-size: 0.95rem; margin-top: 0.2rem; }

    .card {
        background: #171923;
        border: 1px solid #262a37;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        background: #1f2333;
        border: 1px solid #2d3148;
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        margin-right: 0.5rem;
        font-size: 0.8rem;
        color: #c4b5fd;
    }
    .status-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.35rem 0;
        font-size: 0.92rem;
    }
    .stChatMessage { border-radius: 12px; }

    section[data-testid="stSidebar"] {
        background: #12141c;
        border-right: 1px solid #262a37;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #262a37;
        border-radius: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# SESSION STATE
# =========================================================================
defaults = {
    "result": None,          # output dict from run_pipeline
    "processing": False,
    "chat_history": [],      # list of (role, text)
    "history": [],           # past runs: list of dicts {title, time, result}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================================
# SIDEBAR — INPUT CONTROLS
# =========================================================================
with st.sidebar:
    st.markdown("### 🎬 AI Video Assistant")
    st.caption("Turn any video or recording into a searchable, summarized meeting.")

    st.divider()

    input_mode = st.radio("Source type", ["YouTube URL", "Upload file"], horizontal=True)

    source = None
    if input_mode == "YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        source = url.strip() if url else None
    else:
        uploaded = st.file_uploader(
            "Upload audio / video",
            type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
        )
        if uploaded is not None:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = tmp_path
            st.success(f"Loaded: {uploaded.name}")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.divider()

    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing or not source,
    )

    if st.session_state.result:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    if not source and not st.session_state.result:
        st.info("Add a YouTube URL or upload a file to get started.")

    # Past runs, if any
    if st.session_state.history:
        st.divider()
        st.markdown("#### 🕘 Past runs")
        for i, h in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {h['title'][:35]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.result = h["result"]
                st.session_state.chat_history = []
                st.rerun()


# =========================================================================
# HEADER
# =========================================================================
st.markdown('<div class="main-header">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Transcribe, summarize, and chat with any meeting or video</div>',
    unsafe_allow_html=True,
)
st.write("")


# =========================================================================
# PIPELINE EXECUTION — with live, dynamic status
# =========================================================================
if run_clicked and source:
    st.session_state.processing = True
    st.session_state.chat_history = []

    steps = [
        "Loading & chunking source",
        "Transcribing audio",
        "Generating title",
        "Summarizing content",
        "Extracting action items",
        "Extracting key decisions",
        "Extracting open questions",
        "Building chat index (RAG)",
    ]

    status_box = st.status("Running pipeline...", expanded=True)
    progress = st.progress(0)

    try:
        # We call the real run_pipeline() once (it isn't step-callback based),
        # but we animate a matching status list so the UI feels alive while it runs.
        for i, step in enumerate(steps[:-1]):
            status_box.write(f"⏳ {step}...")
            progress.progress(int((i + 1) / len(steps) * 90))
            time.sleep(0.05)  # cosmetic pacing; real work happens in run_pipeline below

        status_box.write("⏳ Running full pipeline (this does the actual work)...")
        result = run_pipeline(source, language)

        progress.progress(100)
        status_box.update(label="✅ Pipeline complete!", state="complete", expanded=False)

        st.session_state.result = result
        st.session_state.history.append(
            {
                "title": result.get("title", "Untitled"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "result": result,
            }
        )

    except Exception as e:
        status_box.update(label="❌ Pipeline failed", state="error", expanded=True)
        st.error(f"Something went wrong: {e}")
        st.session_state.result = None

    st.session_state.processing = False
    st.rerun()


# =========================================================================
# RESULTS DISPLAY
# =========================================================================
result = st.session_state.result

if result:
    st.markdown(f"## 📌 {result.get('title', 'Untitled Meeting')}")

    transcript = result.get("transcript", "") or ""
    word_count = len(transcript.split())
    read_min = max(1, word_count // 200)

    st.markdown(
        f"""
        <span class="metric-pill">🗣️ {word_count:,} words</span>
        <span class="metric-pill">⏱️ ~{read_min} min read</span>
        <span class="metric-pill">🌐 {language}</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "🔑 Decisions", "❓ Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(result.get("summary", "No summary generated."))
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download summary",
            result.get("summary", ""),
            file_name="summary.txt",
            use_container_width=False,
        )

    with tab_transcript:
        with st.expander("Full transcript", expanded=True):
            st.text_area("", transcript, height=400, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript",
            transcript,
            file_name="transcript.txt",
        )

    with tab_actions:
        items = result.get("action_items", "")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if isinstance(items, list):
            for it in items:
                st.checkbox(str(it), key=f"action_{hash(it)}")
        else:
            st.write(items or "No action items found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_decisions:
        decisions = result.get("key_decisions", "")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if isinstance(decisions, list):
            for d in decisions:
                st.markdown(f"- 🔑 {d}")
        else:
            st.write(decisions or "No key decisions found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_questions:
        questions = result.get("open_questions", "")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if isinstance(questions, list):
            for q in questions:
                st.markdown(f"- ❓ {q}")
        else:
            st.write(questions or "No open questions found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_chat:
        st.caption("Ask anything about this meeting — answers are grounded in the transcript.")

        chat_container = st.container(height=430)
        with chat_container:
            for role, text in st.session_state.chat_history:
                with st.chat_message(role):
                    st.write(text)

        question = st.chat_input("Ask a question about this meeting...")
        if question:
            st.session_state.chat_history.append(("user", question))
            rag_chain = result.get("rag_chain")
            try:
                answer = ask_question(rag_chain, question)
            except Exception as e:
                answer = f"⚠️ Couldn't get an answer: {e}"
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 3rem;">
            <h3>👋 No meeting analyzed yet</h3>
            <p style="color:#9ca3af;">
                Add a YouTube URL or upload a file in the sidebar, then hit
                <b>Run Analysis</b> to get a title, summary, action items,
                key decisions, open questions — and a chat interface for the meeting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )