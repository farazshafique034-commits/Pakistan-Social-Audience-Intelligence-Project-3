import streamlit as st
import plotly.express as px
from project3.analytics.dashboard_data import (
    load_or_stop, project3_header, csv_download
)

project3_header(
    "Content & Campaign Intelligence",
    "Project 3 | Social-media engagement and campaign analysis"
)

data = load_or_stop()
posts = data["posts"].copy()

platforms = sorted(posts["platform"].dropna().unique())
selected = st.multiselect(
    "Platforms", platforms, default=platforms
)
posts = posts[posts["platform"].isin(selected)]

types = sorted(posts["content_type"].dropna().unique())
selected_types = st.multiselect(
    "Content types", types, default=types
)
posts = posts[posts["content_type"].isin(selected_types)]

if posts.empty:
    st.warning("No posts match your filters.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Posts", f"{len(posts):,}")
c2.metric("Platforms", posts["platform"].nunique())
c3.metric("Campaigns", posts["campaign"].nunique())

posts["likes_per_day"] = (
    posts["likes"] / posts["days_since_post"].clip(lower=1)
)
posts["comments_per_day"] = (
    posts["comments_count"] / posts["days_since_post"].clip(lower=1)
)

content = posts.groupby(
    ["platform", "content_type"], as_index=False
).agg(
    posts=("post_id", "count"),
    median_likes_per_day=("likes_per_day", "median"),
    median_comments_per_day=("comments_per_day", "median")
)

st.subheader("Content Format Comparison")
fig = px.bar(
    content, x="content_type", y="median_likes_per_day",
    color="platform", barmode="group",
    title="Median Likes Per Day by Content Type"
)
st.plotly_chart(fig, use_container_width=True)

campaign = posts.groupby(
    ["platform", "campaign"], as_index=False
).agg(
    posts=("post_id", "count"),
    median_comments_per_day=("comments_per_day", "median")
)

st.subheader("Campaign Engagement")
fig2 = px.bar(
    campaign, x="campaign", y="median_comments_per_day",
    color="platform", barmode="group",
    title="Median Comments Per Day by Campaign"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Views vs Likes")
available = posts.dropna(subset=["views", "likes"])
if not available.empty:
    fig3 = px.scatter(
        available, x="views", y="likes",
        color="platform", hover_data=["post_id", "content_type"],
        title="Posts with Available View Data"
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("View counts are unavailable for these posts.")

st.subheader("Content Summary")
st.dataframe(content, use_container_width=True, hide_index=True)
csv_download(
    "Export content performance", content,
    "project3_content_intelligence.csv"
)
st.caption("Comparisons are descriptive, not causal or ROAS estimates.")
