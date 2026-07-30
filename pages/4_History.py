import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.db import get_all_analyses, search_bugs

st.set_page_config(page_title="Bug History & Logs - Bug Analyzer", page_icon="📜", layout="wide")

st.title("📜 Bug History & Logs Repository")
st.caption("Search, filter, and inspect historical bug reports and AI diagnosis records")

# Search and Filters
c_search, c_cat, c_sev = st.columns([2, 1, 1])

with c_search:
    search_query = st.text_input("🔍 Search Bugs (Title, Description, Module, Team)", placeholder="Type keywords...")

records = search_bugs(search_query) if search_query else get_all_analyses()

categories = sorted(list(set(r["category"] for r in records))) if records else []
severities = ["All", "Critical", "High", "Medium", "Low"]

with c_cat:
    selected_cat = st.selectbox("Category Filter", ["All"] + categories)

with c_sev:
    selected_sev = st.selectbox("Severity Filter", severities)

# Filter application
filtered = records
if selected_cat != "All":
    filtered = [r for r in filtered if r["category"] == selected_cat]
if selected_sev != "All":
    filtered = [r for r in filtered if r["severity"] == selected_sev]

st.markdown(f"**Found {len(filtered)} records**")

if not filtered:
    st.info("No matching bug records found.")
else:
    for rec in filtered:
        with st.expander(f"Bug #{rec['id']} [{rec['severity']}] - {rec['title']} ({rec['category']})"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Module:**", rec['module'])
                st.write("**Team:**", rec['team'])
            with c2:
                st.write("**Reporter:**", rec['user'])
                st.write("**Date:**", rec['date'])
            with c3:
                st.write("**Confidence:**", f"{int(float(rec['confidence']) * 100)}%")

            st.write("**Description:**", rec['description'])
            st.write("**Explanation:**", rec['explanation'])

            st.write("**Possible Causes:**")
            for cause in rec['causes']:
                st.write(f"- {cause}")

            st.write("**Recommended Tests:**")
            for test in rec['tests']:
                st.write(f"- {test}")
