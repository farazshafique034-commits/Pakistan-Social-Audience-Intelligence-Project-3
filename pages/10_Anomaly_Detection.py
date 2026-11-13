import streamlit as st
import plotly.express as px
from project3.analytics.dashboard_data import (
    load_or_stop, project3_header, csv_download
)

project3_header(
    "AI Anomaly Detection",
    "Project 3 | Isolation Forest and unusual engagement patterns"
)

data = load_or_stop()
posts = data["posts"].copy()

platforms = sorted(posts["platform"].dropna().unique())
selected = st.multiselect(
    "Platform filters", platforms, default=platforms
)
posts = posts[posts["platform"].isin(selected)]

only_flagged = st.checkbox(
    "Show flagged posts only", value=False
)
if only_flagged:
    posts = posts[posts["is_anomaly"] == 1]

if posts.empty:
    st.warning("No posts match the selected filters.")
    st.stop()

flagged = int(posts["is_anomaly"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Posts Displayed", f"{len(posts):,}")
c2.metric("Flagged Posts", flagged)
c3.metric("Flagged Share", f"{100 * flagged / len(posts):.1f}%")

st.subheader("Engagement Anomaly Explorer")
view_data = posts.dropna(subset=["views", "likes"]).copy()
view_data["model_status"] = view_data["is_anomaly"].map(
    {0: "Not flagged", 1: "Flagged"}
)

if not view_data.empty:
    fig = px.scatter(
        view_data, x="views", y="likes",
        color="model_status", hover_data=[
            "post_id", "platform", "content_type", "anomaly_score"
        ],
        title="Likes vs Views: Isolation Forest Results"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No view counts available for this selection.")

st.subheader("Anomaly Score Distribution")
fig2 = px.histogram(
    posts, x="anomaly_score", color="platform",
    nbins=35, barmode="overlay",
    title="Higher Scores Indicate More Unusual Patterns"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Posts Requiring Investigation")
table = posts[posts["is_anomaly"] == 1].sort_values(
    "anomaly_score", ascending=False
)
columns = [
    "post_id", "platform", "content_type", "campaign",
    "likes", "comments_count", "views",
    "anomaly_score", "investigation_reason"
]
st.dataframe(
    table[columns], use_container_width=True, hide_index=True
)
csv_download(
    "Export anomaly investigations", table,
    "project3_anomaly_investigations.csv"
)

st.caption(
    "Anomaly flags are investigation signals, not proof of fraud. "
    "Reasons are descriptive comparisons, not causal explanations."
)
