import hashlib
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from premium_core import ROOT, heading, plot, sankey, text_3d
from premium_pages.p3_csv_engine import analyze_upload


def safe_csv(frame):
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value:
            (" " + value if isinstance(value, str) and
             value.lstrip().startswith(("=", "+", "-", "@")) else value)
        )
    return result.to_csv(index=False).encode("utf-8-sig")


heading(
    "PROJECT 03 / DATA INTELLIGENCE",
    "CSV Upload & AI Analysis Studio",
    "Upload, validate, cluster, detect anomalies and explore interactive visualizations."
)

st.caption(
    "CSV uploads are processed in memory. Your original datasets and ML files stay unchanged."
)

mode = st.radio(
    "Choose data source",
    ["Upload new CSV files", "Use saved Project 3 data"],
    horizontal=True
)

if mode == "Upload new CSV files":
    left_upload, right_upload = st.columns(2)
    with left_upload:
        uploaded_posts = st.file_uploader(
            "Upload Social Posts CSV", type=["csv"], key="p3_new_posts"
        )
    with right_upload:
        uploaded_comments = st.file_uploader(
            "Upload Social Comments CSV", type=["csv"], key="p3_new_comments"
        )

    with st.expander("Required CSV columns and example templates"):
        st.markdown(
            "**Posts:** post_id, platform, likes. "
            "Recommended: comments_count, shares, views, campaign, content_type."
        )
        st.markdown(
            "**Comments:** post_id, comment_text. "
            "Recommended: comment_id, language_type, data_origin."
        )
        example_posts = pd.DataFrame({
            "post_id": ["P001", "P002"],
            "platform": ["Instagram", "TikTok"],
            "campaign": ["11.11 Sale", "Eid Collection"],
            "content_type": ["Reel", "Styling Video"],
            "likes": [1200, 800],
            "comments_count": [2, 1],
            "shares": [65, 32],
            "views": [15000, 9800],
            "data_origin": ["synthetic", "synthetic"]
        })
        example_comments = pd.DataFrame({
            "comment_id": ["C001", "C002", "C003"],
            "post_id": ["P001", "P001", "P002"],
            "comment_text": [
                "Price please?", "Delivery kitne din mein hogi?",
                "Beautiful collection"
            ],
            "language_type": ["English", "Roman Urdu", "English"],
            "data_origin": ["synthetic", "synthetic", "synthetic"]
        })
        a, b = st.columns(2)
        a.download_button(
            "Download Posts Template", safe_csv(example_posts),
            "project3_posts_template.csv", "text/csv"
        )
        b.download_button(
            "Download Comments Template", safe_csv(example_comments),
            "project3_comments_template.csv", "text/csv"
        )

    if uploaded_posts is None or uploaded_comments is None:
        st.info("Upload both Posts and Comments CSV files to start AI analysis.")
        st.stop()

    posts_bytes = uploaded_posts.getvalue()
    comments_bytes = uploaded_comments.getvalue()
    source_label = "Uploaded CSV files"

else:
    folder = ROOT / "project3" / "data" / "processed"
    posts_file = folder / "posts_with_anomalies.csv"
    comments_file = folder / "comments_interpreted.csv"
    if not posts_file.exists() or not comments_file.exists():
        st.error("Saved Project 3 datasets are missing.")
        st.info("Generate the Project 3 dataset and ML results first.")
        st.stop()
    posts_bytes = posts_file.read_bytes()
    comments_bytes = comments_file.read_bytes()
    source_label = "Saved Project 3 analysis"

fingerprint = hashlib.sha256(
    posts_bytes + b"PROJECT3_SEPARATOR" + comments_bytes
).hexdigest()

if st.session_state.get("p3_csv_fingerprint") != fingerprint:
    try:
        with st.spinner("Validating and analyzing Project 3 CSV data..."):
            prepared = analyze_upload(posts_bytes, comments_bytes)
    except (ValueError, TypeError, KeyError) as exc:
        st.error("CSV processing failed: " + str(exc))
        st.stop()
    st.session_state["p3_csv_fingerprint"] = fingerprint
    st.session_state["p3_csv_results"] = prepared

posts, comments, notes = st.session_state["p3_csv_results"]
posts = posts.copy()
comments = comments.copy()

st.success("Project 3 CSV analysis is ready: " + source_label)

if posts["data_origin"].eq("unspecified").any():
    st.warning("Some uploaded records have unspecified data provenance.")

for note in notes:
    st.caption(note)

st.divider()

platforms = sorted(posts["platform"].dropna().unique())
selected = st.multiselect(
    "Filter social platforms", platforms, default=platforms
)
filtered_posts = posts[posts["platform"].isin(selected)].copy()
filtered_comments = comments[
    comments["post_id"].isin(filtered_posts["post_id"])
].copy()

if filtered_posts.empty:
    st.info("Select at least one platform.")
    st.stop()

filtered_comments["cluster_name"] = (
    "Cluster " + filtered_comments["cluster_id"].astype(str)
)
filtered_posts["anomaly_status"] = filtered_posts["review_status"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Social Posts", f"{len(filtered_posts):,}")
c2.metric("Audience Comments", f"{len(filtered_comments):,}")
c3.metric("Discovered Clusters", filtered_comments["cluster_id"].nunique())
c4.metric("Flagged Posts", int(filtered_posts["is_anomaly"].sum()))

tab1, tab2, tab3, tab4 = st.tabs([
    "Audience Clusters",
    "3D Engagement",
    "Content Intelligence",
    "Explore & Export"
])

with tab1:
    st.subheader("3D Audience Text Clusters")
    if len(filtered_comments) >= 5:
        try:
            figure = text_3d(
                filtered_comments, "clean_text", "cluster_name",
                "Audience Clusters - 3D Text Similarity"
            )
            if figure is not None:
                plot(figure, height=600)
            else:
                st.info("Not enough distinct text features for a 3D projection.")
        except ValueError as exc:
            st.info("3D visualization unavailable: " + str(exc))

    st.subheader("Audience Cluster Sizes")
    counts = (
        filtered_comments.groupby(
            ["cluster_name", "cluster_label"], as_index=False
        ).size().rename(columns={"size": "comments"})
    )
    if not counts.empty:
        figure = px.bar(
            counts, x="cluster_name", y="comments",
            color="cluster_name", text_auto=True,
            title="Discovered Audience Segment Distribution"
        )
        plot(figure)
        st.dataframe(counts, use_container_width=True, hide_index=True)

    st.subheader("Cluster to Language Flow")
    if not filtered_comments.empty:
        figure = sankey(
            filtered_comments, "cluster_name", "language_type",
            title="Audience Clusters and Comment Languages"
        )
        plot(figure, height=520)

with tab2:
    st.subheader("3D Engagement Anomaly Explorer")
    available = filtered_posts.dropna(
        subset=["likes", "comments_count", "views"]
    )
    if not available.empty:
        figure = px.scatter_3d(
            available, x="likes", y="comments_count", z="views",
            color="anomaly_status",
            hover_data=["post_id", "platform", "campaign"],
            opacity=0.75,
            title="Likes, Comments and Views - 3D Visualization"
        )
        plot(figure, height=610)
    else:
        st.info("View counts are not available for the selected posts.")

    if filtered_posts["anomaly_score"].notna().any():
        figure = px.histogram(
            filtered_posts, x="anomaly_score",
            color="anomaly_status", nbins=35,
            title="Isolation Forest Anomaly Scores"
        )
        plot(figure)
    st.dataframe(
        filtered_posts[filtered_posts["is_anomaly"].eq(1)],
        use_container_width=True, hide_index=True
    )

with tab3:
    st.subheader("Content Hierarchy - Sunburst")
    eligible = filtered_posts[
        filtered_posts["likes"].gt(0)
    ].copy()
    if not eligible.empty:
        figure = px.sunburst(
            eligible,
            path=["platform", "campaign", "content_type"],
            values="likes",
            title="Platform / Campaign / Content Type"
        )
        plot(figure, height=580)

    st.subheader("Content-Type Performance")
    summary = filtered_posts.groupby(
        ["platform", "content_type"], as_index=False
    ).agg(
        posts=("post_id", "count"),
        median_likes=("likes", "median"),
        median_comments=("comments_count", "median")
    )
    figure = px.bar(
        summary, x="content_type", y="median_likes",
        color="platform", barmode="group",
        title="Median Likes by Content Type"
    )
    plot(figure)
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Explore Uploaded Data")
    show = st.radio(
        "Choose table", ["Posts", "Comments"], horizontal=True
    )
    frame = filtered_posts if show == "Posts" else filtered_comments
    st.dataframe(frame.head(1000), use_container_width=True)

    left, right = st.columns(2)
    left.download_button(
        "Download Analyzed Posts CSV",
        safe_csv(filtered_posts),
        "project3_analyzed_posts.csv",
        mime="text/csv"
    )
    right.download_button(
        "Download Clustered Comments CSV",
        safe_csv(filtered_comments),
        "project3_clustered_comments.csv",
        mime="text/csv"
    )

st.caption(
    "Cluster descriptions are exploratory. Anomalies require manual review. "
    "Synthetic data cannot establish real-world marketing performance."
)
