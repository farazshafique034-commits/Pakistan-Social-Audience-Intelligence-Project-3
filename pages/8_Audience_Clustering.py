import streamlit as st
import plotly.express as px
from project3.analytics.dashboard_data import (
    load_or_stop, project3_header, csv_download
)

project3_header(
    "Audience Clustering Intelligence",
    "Project 3 | TF-IDF, K-Means and discovered audience groups"
)

data = load_or_stop()
comments = data["comments"].copy()
posts = data["posts"]

platform_options = sorted(posts["platform"].dropna().unique())
selected = st.multiselect(
    "Filter platforms", platform_options, default=platform_options
)

selected_ids = set(posts.loc[
    posts["platform"].isin(selected), "post_id"
])
comments = comments[comments["post_id"].isin(selected_ids)]

if comments.empty:
    st.warning("No audience comments for the selected filters.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Comments", f"{len(comments):,}")
c2.metric("Discovered Groups", comments["cluster_id"].nunique())
c3.metric("Linked Posts", comments["post_id"].nunique())

summary = (
    comments.groupby(["cluster_id", "cluster_label"], as_index=False)
    .size().rename(columns={"size": "comments"})
)
summary["cluster_name"] = summary["cluster_id"].apply(
    lambda value: f"Cluster {int(value)}"
)

st.subheader("Audience Segment Distribution")
fig = px.bar(
    summary, x="cluster_name", y="comments",
    color="cluster_name", text_auto=True,
    title="K-Means Cluster Sizes"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Audience Groups and Keywords")
st.dataframe(
    summary[["cluster_id", "cluster_label", "comments"]],
    use_container_width=True, hide_index=True
)

st.subheader("Language Distribution")
language = (
    comments.groupby(["cluster_id", "language_type"], as_index=False)
    .size().rename(columns={"size": "comments"})
)
fig2 = px.bar(
    language, x="cluster_id", y="comments",
    color="language_type", barmode="stack"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Explore Representative Comments")
cluster_ids = sorted(comments["cluster_id"].dropna().unique())
choice = st.selectbox(
    "Select cluster", cluster_ids, format_func=lambda x: f"Cluster {int(x)}"
)
examples = comments[comments["cluster_id"] == choice]
st.dataframe(
    examples[["comment_text", "language_type", "cluster_label"]].head(100),
    use_container_width=True, hide_index=True
)

csv_download(
    "Export filtered clustering data", comments,
    "project3_audience_clusters.csv"
)
st.caption("Cluster labels are exploratory, not confirmed customer intents.")
