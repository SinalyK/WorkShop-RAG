import streamlit as st

from app.workshop_ import agent
import asyncio

# 

st.set_page_config(page_title="RAG Intelligent Agent", page_icon="📚", layout="wide")

# ------------------ Styling ------------------
st.markdown(
    """
    <style>
    body { background-color: #f6f8fb; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .header {
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        padding: 1.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 2rem;
    }

    .chat-user {
        background: #e0e7ff;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    .chat-agent {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
    }

    .source-box {
        background: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Header ------------------
st.markdown(
    """
    <div class="header">
        <h1>📚 Retrieval‑Augmented Generation Agent</h1>
        <p>Ask questions. The agent retrieves relevant documents and generates grounded answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Agent Settings")
st.sidebar.markdown("This agent decides when to search documents and when to answer.")


top_k = st.sidebar.slider("Retrieved Chunks (top‑k)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.caption("RAG • Vector Search • LLM")

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Input ------------------
user_question = st.text_input(
    "💬 Ask a question about your documents",
    placeholder="e.g. What does the document say about retrieval‑augmented generation?",
)

if st.button("🚀 Ask the Agent") and user_question:
    with st.spinner("Agent is reasoning and searching..."):
        result = asyncio.run(agent.arun(user_question))

        answer = result.get("final_answer", "Sorry, I could not find an answer.")
        sources = ["Document chunk 1", "Document chunk 2", "Document chunk 3"]

        st.session_state.chat_history.append(
            {"question": user_question, "answer": answer, "sources": sources}
        )

# ------------------ Chat History ------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown(
        f"<div class='chat-user'><b>🧑 User</b><br>{chat['question']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='chat-agent'><b>🤖 RAG Agent</b><br>{chat['answer']}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Retrieved Sources"):
        for src in chat["sources"]:
            st.markdown(
                f"<div class='source-box'>• {src}</div>", unsafe_allow_html=True
            )

# ------------------ Footer ------------------
st.markdown(
    "<div class='footer'>RAG Intelligent Agent • Document Retrieval + Grounded Generation</div>",
    unsafe_allow_html=True,
)
