import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.db import get_all_analyses, get_bug_by_id

st.set_page_config(page_title="Bug Analysis Output - Bug Analyzer", page_icon="📊", layout="wide")

st.title("📊 Detailed Bug Analysis Output")

records = get_all_analyses()

if not records:
    st.info("No bug analyses found in database. Go to '1_Input' page to submit your first bug report.")
else:
    bug_options = {f"Bug #{r['id']} - {r['title']} ({r['date']})": r['id'] for r in records}
    selected_option = st.selectbox("Select Bug Analysis Record:", list(bug_options.keys()))
    selected_id = bug_options[selected_option]

    rec = get_bug_by_id(selected_id)

    if rec:
        st.markdown(f"### Bug #{rec['id']}: {rec['title']}")
        st.caption(f"Reported by: **{rec['user']}** on {rec['date']} | Created At: {rec['created_at']}")
        st.write("**Description:**", rec['description'])

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Category", rec["category"])
        with c2:
            st.metric("Severity", rec["severity"])
        with c3:
            st.metric("Affected Module", rec["module"])
        with c4:
            st.metric("Assigned Team", rec["team"])

        conf = float(rec.get("confidence", 0.85))
        st.markdown(f"**Confidence Score:** {int(conf * 100)}%")
        st.progress(conf)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🔍 Possible Root Causes")
            for c in rec["causes"]:
                st.markdown(f"- {c}")

            st.markdown("#### 🧠 Technical Explanation")
            st.info(rec["explanation"])

        with col_right:
            st.markdown("#### ✅ Recommended QA Test Suite")
            for idx, test_step in enumerate(rec["tests"], 1):
                st.checkbox(f"{idx}. {test_step}", key=f"output_test_{rec['id']}_{idx}")

        with st.expander("🔬 NLP Feature Extraction Analysis"):
            st.write("**Extracted Tokens:**", rec["tokens"])
            st.write("**Extracted Keyphrases:**", rec["keyphrases"])
