import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

def prepare_analytics_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts database records list into a pandas DataFrame."""
    if not records:
        return pd.DataFrame(columns=[
            "id", "title", "user", "date", "category", "severity",
            "module", "team", "confidence", "created_at"
        ])
    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def build_category_bar_chart(df: pd.DataFrame):
    """Builds a bar chart for bug category distribution."""
    if df.empty:
        return px.bar(title="Category Distribution (No Data)")

    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        color="Category",
        text="Count",
        title="Bug Distribution by Category",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(showlegend=False, template="plotly_dark", height=380)
    return fig

def build_severity_pie_chart(df: pd.DataFrame):
    """Builds a pie/donut chart for bug severity breakdown."""
    if df.empty:
        return px.pie(title="Severity Breakdown (No Data)")

    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]

    color_map = {
        "Critical": "#EF5350",
        "High": "#FFA726",
        "Medium": "#42A5F5",
        "Low": "#66BB6A"
    }

    fig = px.pie(
        counts,
        names="Severity",
        values="Count",
        title="Bug Severity Breakdown",
        hole=0.4,
        color="Severity",
        color_discrete_map=color_map
    )
    fig.update_layout(template="plotly_dark", height=380)
    return fig

def build_daily_timeline_chart(df: pd.DataFrame):
    """Builds a line timeline chart of bug reports per day."""
    if df.empty:
        return px.line(title="Bug Reports Timeline (No Data)")

    daily = df.groupby("date").size().reset_index(name="Count")
    fig = px.line(
        daily,
        x="date",
        y="Count",
        markers=True,
        title="Daily Bug Submission Timeline",
        line_shape="spline"
    )
    fig.update_layout(template="plotly_dark", height=350, xaxis_title="Date", yaxis_title="Submissions")
    return fig

def build_module_bar_chart(df: pd.DataFrame):
    """Builds horizontal bar chart for top affected modules."""
    if df.empty:
        return px.bar(title="Affected Modules (No Data)")

    counts = df["module"].value_counts().head(8).reset_index()
    counts.columns = ["Module", "Count"]

    fig = px.bar(
        counts,
        y="Module",
        x="Count",
        orientation="h",
        color="Count",
        title="Top Affected System Modules",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_dark", height=380, yaxis={"categoryorder": "total ascending"})
    return fig

def build_team_donut_chart(df: pd.DataFrame):
    """Builds donut chart for suggested team assignment distribution."""
    if df.empty:
        return px.pie(title="Team Assignments (No Data)")

    counts = df["team"].value_counts().reset_index()
    counts.columns = ["Team", "Count"]

    fig = px.pie(
        counts,
        names="Team",
        values="Count",
        title="Assigned Engineering Teams",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(template="plotly_dark", height=380)
    return fig
