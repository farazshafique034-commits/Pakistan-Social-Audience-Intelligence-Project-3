import pandas as pd
import plotly.express as px
import streamlit as st
from premium_core import ROOT, heading, plot, sankey, text_3d

heading(
    "PROJECT 03 / AI SOCIAL INTELLIGENCE",
    "Audience & Campaign Intelligence Studio",
    "3D audience clusters, engagement anomalies and marketing opportunities."
)

base = ROOT / "project3" / "data" / "processed"
posts_path = base / "posts_with_anomalies.csv"
comments_path = base / "comments_interpreted.csv"

if not posts_path.exists() or not comments_path.exists():
    st.error("Project 3 processed datasets are missing.")
    st.code("python -m project3.pipelines.generate_dataset")
    st.info(
        "Run the Project 3 processing and ML pipeline to generate dashboard data."
    )
    st.stop()

posts = pd.read_csv(posts_path, encoding="utf-8-sig")
comments = pd.read_csv(comments_path, encoding="utf-8-sig")

if posts.empty or comments.empty:
    st.error("Project 3 datasets are empty.")
    st.stop()

st.warning(
    "SYNTHETIC DATA: These results are a controlled class demonstration, "
    "not measured performance of Pakistani fashion brands."
)

platforms = sorted(posts["platform"].dropna().astype(str).unique())
selected = st.multiselect(
    "Select Social Media Platforms", platforms, default=platforms
)
posts = posts[posts["platform"].isin(selected)].copy()
comments = comments[comments["post_id"].isin(posts["post_id"])].copy()

if posts.empty or comments.empty:
    st.info("No matching social media records.")
    st.stop()

comments["cluster_name"] = comments["cluster_id"].apply(
    lambda x: f"Cluster {int(x)}"
)
posts["anomaly_status"] = posts["is_anomaly"].map(
    {0: "Normal", 1: "Flagged"}
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Social Posts", f"{len(posts):,}")
c2.metric("Audience Comments", f"{len(comments):,}")
c3.metric("Audience Clusters", comments["cluster_id"].nunique())
c4.metric("Flagged Anomalies", int(posts["is_anomaly"].sum()))

st.subheader("3D Audience Clustering Space")
try:
    fig = text_3d(
        comments, "clean_text", "cluster_name",
        "Audience Clusters: TF-IDF + 3D Text Projection"
    )
    if fig is not None:
        plot(fig, height=620)
    else:
        st.info("Insufficient unique text features for 3D visualization.")
except ValueError as exc:
    st.info("3D clustering chart unavailable: " + str(exc))
st.caption(
    "Projection is for visualization. Original cluster IDs come from the trained K-Means pipeline."
)

st.subheader("3D Engagement Anomaly Explorer")
available = posts.dropna(
    subset=["likes", "comments_count", "views"]
).copy()
if not available.empty:
    fig = px.scatter_3d(
        available, x="likes", y="comments_count", z="views",
        color="anomaly_status",
        hover_data=["post_id", "platform", "content_type", "anomaly_score"],
        opacity=0.75,
        title="3D Engagement: Likes / Comments / Views"
    )
    fig.update_traces(marker=dict(size=4))
    plot(fig, height=600)
else:
    st.info("No usable view counts for 3D anomaly analysis.")

left, right = st.columns(2)
with left:
    st.subheader("Campaign Content Sunburst")
    sunburst_data = posts.dropna(
        subset=["platform", "campaign", "content_type", "likes"]
    )
    fig = px.sunburst(
        sunburst_data,
        path=["platform", "campaign", "content_type"],
        values="likes",
        title="Platform / Campaign / Content Type"
    )
    plot(fig, height=570)

with right:
    st.subheader("Audience Cluster Distribution")
    grouped = comments.groupby(
        "cluster_name", as_index=False
    ).size().rename(columns={"size": "comments"})
    fig = px.treemap(
        grouped, path=["cluster_name"], values="comments",
        color="cluster_name", title="Discovered Audience Segments"
    )
    plot(fig, height=570)

st.subheader("Audience Segments and Languages")
plot(sankey(
    comments, "cluster_name", "language_type",
    title="Audience Cluster to Comment Language"
), height=500)

st.subheader("Audience and Post Investigation")
tab1, tab2 = st.tabs(["Audience Comments", "Anomalous Posts"])
with tab1:
    columns = [
        c for c in [
            "comment_text", "language_type", "cluster_id", "cluster_label"
        ] if c in comments.columns
    ]
    st.dataframe(
        comments[columns].head(300), use_container_width=True
    )
with tab2:
    flagged = posts[posts["is_anomaly"] == 1].sort_values(
        "anomaly_score", ascending=False
    )
    st.dataframe(
        flagged.head(200), use_container_width=True
    )

st.download_button(
    "Export filtered audience intelligence",
    comments.to_csv(index=False).encode("utf-8-sig"),
    "project3_audience_intelligence.csv", "text/csv"
)
