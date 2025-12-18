import streamlit as st
from workshop import agent
import asyncio

st.set_page_config(
    page_title="RAG Intelligent Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ Ultra Modern Glassmorphic Theme ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Dark Theme with Animated Background */
    [data-testid="stAppViewContainer"] {
        background: #0a0e27;
        background-image: 
            radial-gradient(at 47% 33%, hsl(240, 63%, 13%) 0, transparent 59%), 
            radial-gradient(at 82% 65%, hsl(218, 100%, 35%) 0, transparent 55%),
            radial-gradient(at 15% 75%, hsl(280, 75%, 25%) 0, transparent 50%);
        color: #e0e6ff;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            linear-gradient(45deg, transparent 30%, rgba(59, 130, 246, 0.05) 50%, transparent 70%),
            linear-gradient(-45deg, transparent 30%, rgba(139, 92, 246, 0.05) 50%, transparent 70%);
        animation: backgroundShift 15s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(5%, 5%) rotate(180deg); }
    }
    
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        border-right: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(20px);
    }
    
    /* Main Container with Glassmorphism */
    .main-container {
        background: rgba(15, 20, 45, 0.7);
        border-radius: 30px;
        margin: 0;
        padding: 0;
        backdrop-filter: blur(40px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 30px 60px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        z-index: 1;
    }
    
    /* Floating Header with Gradient */
    .hero-header {
        text-align: center;
        padding: 60px 40px 40px;
        position: relative;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 30px 30px 0 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        animation: shimmer 4s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 200%; }
    }
    
    .hero-title {
        font-size: 4.5em;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -3px;
        text-shadow: 0 0 60px rgba(96, 165, 250, 0.3);
        position: relative;
        z-index: 1;
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .hero-subtitle {
        font-size: 1.4em;
        color: #94a3b8;
        margin: 15px 0 0;
        font-weight: 400;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 10px 20px;
        border-radius: 25px;
        margin-top: 20px;
        font-size: 0.9em;
        color: #60a5fa;
        backdrop-filter: blur(10px);
    }
    
    /* Floating Input Container */
    .input-container {
        padding: 40px;
        position: relative;
    }
    
    .input-wrapper {
        background: rgba(15, 20, 45, 0.9);
        border-radius: 25px;
        padding: 8px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(59, 130, 246, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .input-wrapper::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
        animation: inputShimmer 3s infinite;
    }
    
    @keyframes inputShimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .input-wrapper:hover, .input-wrapper:focus-within {
        border-color: #3b82f6;
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.5),
            0 0 80px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Chat Container */
    .chat-container {
        padding: 20px 40px 40px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Message Cards with 3D Effect */
    .message-card {
        margin: 25px 0;
        animation: messageSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateY(30px) rotateX(-10deg);
        }
        to {
            opacity: 1;
            transform: translateY(0) rotateX(0);
        }
    }
    
    .user-msg {
        display: flex;
        justify-content: flex-end;
        gap: 15px;
        align-items: flex-start;
    }
    
    .agent-msg {
        display: flex;
        justify-content: flex-start;
        gap: 15px;
        align-items: flex-start;
    }
    
    .avatar {
        width: 45px;
        height: 45px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .avatar::before {
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: 15px;
        padding: 2px;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        animation: avatarGlow 2s ease-in-out infinite;
    }
    
    @keyframes avatarGlow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .agent-avatar {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 25px;
        border-radius: 20px;
        position: relative;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .message-bubble:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.3) 100%);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 20px 20px 5px 20px;
    }
    
    .agent-bubble {
        background: rgba(15, 20, 45, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px 20px 20px 5px;
    }
    
    .message-role {
        font-weight: 700;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        opacity: 0.8;
    }
    
    .user-role {
        color: #60a5fa;
    }
    
    .agent-role {
        color: #a78bfa;
    }
    
    .message-text {
        font-size: 1.05em;
        line-height: 1.7;
        color: #e0e6ff;
    }
    
    /* Sources with Card Design */
    .sources-section {
        margin-top: 15px;
        padding: 20px;
        background: rgba(10, 14, 39, 0.6);
        border-radius: 15px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .sources-title {
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 12px;
        font-size: 0.95em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .source-card {
        background: rgba(139, 92, 246, 0.1);
        border-left: 3px solid #8b5cf6;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        color: #c4b5fd;
        font-size: 0.9em;
        transition: all 0.3s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .source-card:hover {
        background: rgba(139, 92, 246, 0.2);
        transform: translateX(8px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.2);
    }
    
    /* Button with Modern Design */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0px 32px !important;
        border-radius: 15px !important;
        font-size: 1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
        height: 54px !important;
        min-height: 54px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 50%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    /* Text Input Modern Style */
    .stTextInput>div>div>input {
        background: rgba(20, 25, 50, 0.8) !important;
        border: 2px solid rgba(0, 212, 255, 0.4) !important;
        border-radius: 15px !important;
        padding: 16px 24px !important;
        color: #ffffff !important;
        font-size: 1.05em !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        caret-color: #ffffff !important;s
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #666666 !important;
    }
    
    .stTextInput>div>div>input:focus {
        background: rgba(20, 25, 50, 0.95) !important;
        border-color: #00d4ff !important;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.4) !important;
        outline: none !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }
    
    .stTextInput input {
        color: #ffffff !important;
    }
    
    /* Expander Modern */
    .streamlit-expanderHeader {
        background: rgba(15, 20, 45, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        color: #e0e6ff;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(15, 20, 45, 0.8);
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    /* Floating Stats */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        padding: 30px 40px;
        flex-wrap: wrap;
    }
    
    .stat-card {
        background: rgba(15, 20, 45, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 25px 35px;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        transition: all 0.4s ease;
        min-width: 180px;
    }
    
    .stat-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.3);
        border-color: #3b82f6;
    }
    
    .stat-icon {
        font-size: 2.5em;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 20px currentColor);
    }
    
    .stat-value {
        font-size: 2em;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Spinner Modern */
    .stSpinner > div {
        border-color: #3b82f6 transparent transparent transparent !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 20, 45, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5em; }
        .hero-subtitle { font-size: 1.1em; }
        .message-bubble { max-width: 85%; }
        .chat-container { padding: 20px 15px; }
        .input-container { padding: 20px 15px; }
        .stats-container { gap: 15px; }
        .stat-card { min-width: 140px; padding: 20px 25px; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Hero Header ------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">
            <i class="fas fa-brain"></i>
            RAG Agent
        </div>
        <div class="hero-subtitle">
            <i class="fas fa-wifi"></i> Acces internet 
            <i class="fas fa-search" style="margin: 0 10px;"></i> Recherche Documentaire 
            <i class="fas fa-cloud-sun" style="margin: 0 10px;"></i> Acces météorologique
        </div>
        <div class="hero-badge">
            <i class="fas fa-bolt"></i>
            Workshop IA Avancée - S5 - ENIAD
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Stats Section ------------------

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### <i class='fas fa-cog'></i> Paramètres", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### <i class='fas fa-magic'></i> Capacités de l'Agent", unsafe_allow_html=True)
    st.markdown("""
    - <i class='fas fa-search'></i> Recherche Documentaire
    - <i class='fas fa-brain'></i> Acces internet
    - <i class='fas fa-layer-group'></i> Acces météorologique
    - <i class='fas fa-language'></i> Traitement Linguistique
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### <i class='fas fa-info-circle'></i> À Propos", unsafe_allow_html=True)
    st.markdown("*Système RAG de nouvelle génération*")

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Input Section ------------------
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    user_question = st.text_input(
        "",
        placeholder="Posez-moi n'importe quelle question...",
        label_visibility="collapsed",
        key="user_input"
    )
with col2:
    ask_button = st.button("Envoyer", use_container_width=True, key="send_btn")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Processing ------------------
if ask_button and user_question:
    with st.spinner("Traitement de votre question..."):
        result = asyncio.run(agent.arun(user_question))
        answer = result.get("final_answer", "Désolé, je n'ai pas pu trouver de réponse.")
        sources = ["Document chunk 1", "Document chunk 2", "Document chunk 3"]
        st.session_state.chat_history.append(
            {"question": user_question, "answer": answer, "sources": sources}
        )

# ------------------ Chat History ------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat in reversed(st.session_state.chat_history):
    # User message
    st.markdown(
        f"""
        <div class="message-card">
            <div class="user-msg">
                <div class="message-bubble user-bubble">
                    <div class="message-role user-role">
                        <i class="fas fa-user"></i> Vous
                    </div>
                    <div class="message-text">{chat['question']}</div>
                </div>
                <div class="avatar user-avatar">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Agent message container start
    st.markdown(
        """
        <div class="message-card">
            <div class="agent-msg">
                <div class="avatar agent-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-bubble agent-bubble">
                    <div class="message-role agent-role">
                        <i class="fas fa-brain"></i> RAG Agent
                    </div>
                    <div class="message-text">
        """,
        unsafe_allow_html=True,
    )
    
    # Display markdown content
    st.write(chat['answer'])
    
    # Agent message container end
    st.markdown(
        """
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sources
    with st.expander("Sources Récupérées", expanded=False):
        st.markdown(
            f"""
            <div class="sources-section">
                <div class="sources-title">
                    <i class="fas fa-book"></i>
                    Documents de référence
                </div>
                {''.join([f'<div class="source-card"><i class="fas fa-file-alt"></i> {src}</div>' for src in chat["sources"]])}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)