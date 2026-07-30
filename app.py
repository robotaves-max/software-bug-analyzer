import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Page configuration
st.set_page_config(
    page_title="Software Bug Analyzer",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & dark mode compatibility
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .badge-critical {
        background-color: #EF4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #F59E0B;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #3B82F6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar header
st.sidebar.image("https://img.icons8.com/isometric-folders/100/bug.png", width=70)
st.sidebar.title("Bug Analyzer AI")
st.sidebar.caption("NLP + RAG + LLM System v1.0")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("""
- 📝 **1_Input**: Submit Bug Reports
- 📊 **2_Output**: View Bug Analysis
- 📈 **3_Dashboard**: Analytics & Metrics
- 📜 **4_History**: Bug History & Logs
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 Person 1 AI Pipeline runs in the background powered by spaCy, SentenceTransformers, FAISS, and Groq/OpenAI LLM models.")

# Main Landing Content
st.markdown('<div class="main-title">Software Bug Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated NLP Preprocessing, FAISS Semantic RAG Retrieval, and LLM Bug Diagnosis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🔍 6-Stage NLP Pipeline</h3>
        <p>Text sanitization, log removal, spaCy tokenization, custom software entity extraction, and hybrid keyphrase extraction.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ FAISS Vector RAG</h3>
        <p>Semantic vector search over thousands of historical GitHub bugs to construct precise internal reasoning context.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 LLM Bug Diagnosis</h3>
        <p>Predicts Category, Severity, Affected Module, Root Causes, Team Assignment, and Recommended QA Test Suites.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Quick Start")
st.write("Select **1_Input** from the left sidebar to submit a new bug report for AI analysis, or explore **3_Dashboard** to view aggregated team metrics.")
