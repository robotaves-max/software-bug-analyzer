import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.db import get_all_analyses, get_stats
from backend.analytics import (
    prepare_analytics_dataframe,
    build_category_bar_chart,
    build_severity_pie_chart,
    build_daily_timeline_chart,
    build_module_bar_chart,
    build_team_donut_chart,
    generate_pdf_report
)

st.set_page_config(page_title="Analytics Dashboard - Bug Analyzer", page_icon="📈", layout="wide")

st.title("📈 Bug Analytics & Insights Dashboard")
st.caption("Real-time telemetry and metrics across analyzed software bugs")

records = get_all_analyses()
stats = get_stats()

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Bugs Analyzed", stats["total_bugs"])
with c2:
    st.metric("Most Common Category", stats["most_common_category"])
with c3:
    st.metric("Critical Severity Bugs", stats["critical_count"])
with c4:
    st.metric("Recent Submissions", stats["recent_count"])

st.markdown("---")

df = prepare_analytics_dataframe(records)

if df.empty:
    st.info("No analytics data available yet. Submit bugs in '1_Input' page to populate charts.")
else:
    # Row 1 Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_cat = build_category_bar_chart(df)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        fig_sev = build_severity_pie_chart(df)
        st.plotly_chart(fig_sev, use_container_width=True)

    # Row 2 Charts
    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        fig_mod = build_module_bar_chart(df)
        st.plotly_chart(fig_mod, use_container_width=True)

    with col_chart4:
        fig_team = build_team_donut_chart(df)
        st.plotly_chart(fig_team, use_container_width=True)

    # Timeline Chart
    fig_time = build_daily_timeline_chart(df)
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # CSV Download Export
    st.subheader("📥 Export Dataset")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Bug Analytics Report (CSV)",
        data=csv_data,
        file_name="bug_analytics_report.csv",
        mime="text/csv",
        type="primary"
    )

    try:
        pdf_bytes = generate_pdf_report(df)
        st.download_button(
            label="Download Bug Analytics Report (PDF)",
            data=pdf_bytes,
            file_name="bug_analytics_report.pdf",
            mime="application/pdf",
            type="secondary"
        )
    except Exception as exc:
        st.info("PDF export is available when reportlab is installed. Install reportlab to enable PDF downloads.")
