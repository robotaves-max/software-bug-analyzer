import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.ai_service import AIService

st.set_page_config(page_title="Submit Bug Report - Bug Analyzer", page_icon="📝", layout="wide")

@st.cache_resource
def get_ai_service():
    """Instantiates cached singleton AI Service."""
    return AIService()

st.title("📝 Submit Bug Report for AI Analysis")
st.caption("Enter bug details below. The NLP pipeline will extract entities, retrieve internal RAG context, and generate diagnosis.")

with st.form("bug_submission_form"):
    col_title, col_user = st.columns([3, 1])

    with col_title:
        title_input = st.text_input(
            "Bug Title",
            placeholder="e.g., Application crashes while uploading large profile pictures >20MB",
            help="Provide a concise title describing the bug."
        )

    with col_user:
        user_input = st.text_input("Reporter Name", value="Senior Developer")

    desc_input = st.text_area(
        "Bug Description & Stack Trace Logs",
        height=180,
        placeholder="e.g., Uploading images larger than 20MB causes the application to terminate unexpectedly with an OutOfMemory error in image_uploader.ts on line 88 under heavy load.",
        help="Include full error descriptions, code paths, or logs."
    )

    submit_button = st.form_submit_button("🚀 Analyze Bug with AI", type="primary", use_container_width=True)

if submit_button:
    if not title_input.strip() or not desc_input.strip():
        st.error("Please enter both Bug Title and Description before analyzing.")
    else:
        ai_service = get_ai_service()

        with st.spinner("AI analyzing (NLP Processing -> Sentence Embedding -> FAISS Context -> LLM Diagnosis)..."):
            result = ai_service.analyze_bug(title_input, desc_input, user=user_input)

        st.session_state["last_analysis"] = result
        st.success(f"✓ Analysis complete! Saved to database as Bug ID #{result['bug_id']}")

        # Render Diagnosis Visualizer
        st.markdown("---")
        st.subheader("🎯 AI Diagnosis & Recommendation")

        pred = result["prediction"]
        nlp = result["nlp"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Predicted Category", pred.get("category", "N/A"))
        with c2:
            sev = pred.get("severity", "Medium")
            st.metric("Severity Level", sev)
        with c3:
            st.metric("Affected Module", pred.get("affected_module", "N/A"))
        with c4:
            st.metric("Suggested Team", pred.get("suggested_team", "N/A"))

        # Confidence Bar
        conf = float(pred.get("confidence_score", 0.85))
        st.markdown(f"**Confidence Score:** {int(conf * 100)}%")
        st.progress(conf)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🔍 Possible Root Causes")
            causes = pred.get("possible_causes", [])
            for c in causes:
                st.markdown(f"- {c}")

            st.markdown("#### 🧠 Technical Explanation")
            st.info(pred.get("explanation", ""))

        with col_right:
            st.markdown("#### ✅ Recommended Test Suite (Interactive Checkbox)")
            tests = pred.get("recommended_tests", [])
            for idx, test_step in enumerate(tests, 1):
                st.checkbox(f"{idx}. {test_step}", key=f"test_input_{idx}")

        with st.expander("🔬 View Extracted NLP Metadata"):
            st.write("**Cleaned Text:**", nlp.get("clean_text", ""))
            st.write("**Extracted Tokens:**", nlp.get("tokens", []))
            st.write("**Extracted Entities:**", nlp.get("entities", {}))
            st.write("**Extracted Keyphrases:**", nlp.get("keyphrases", []))
