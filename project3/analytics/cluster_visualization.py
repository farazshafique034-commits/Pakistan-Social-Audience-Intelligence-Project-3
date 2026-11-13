import json
import pandas as pd
import plotly.express as px
from project3.config.settings import PROCESSED_DIR, REPORT_DIR

def build_cluster_charts():
    source = PROCESSED_DIR / "comments_interpreted.csv"
    if not source.exists():
        raise FileNotFoundError("Run Phase 4 Part 2 first.")

    data = pd.read_csv(source, encoding="utf-8-sig")
    valid = data[data["cluster_id"] >= 0].copy()
    if valid.empty:
        raise ValueError("No assigned clusters found.")

    valid["cluster"] = valid["cluster_id"].apply(lambda x: f"Cluster {int(x)}")
    folder = REPORT_DIR / "charts"
    folder.mkdir(parents=True, exist_ok=True)

    counts = (
        valid.groupby("cluster", as_index=False)
        .size().rename(columns={"size": "comments"})
    )

    fig1 = px.bar(
        counts, x="cluster", y="comments",
        color="cluster",
        title="Audience Cluster Sizes - Synthetic Dataset",
        text_auto=True
    )
    fig1.write_html(
        str(folder / "audience_cluster_sizes.html"),
        include_plotlyjs=True
    )

    languages = (
        valid.groupby(["cluster", "language_type"], as_index=False)
        .size().rename(columns={"size": "comments"})
    )

    fig2 = px.bar(
        languages, x="cluster", y="comments",
        color="language_type", barmode="stack",
        title="Comment Languages Across Clusters"
    )
    fig2.write_html(
        str(folder / "cluster_languages.html"),
        include_plotlyjs=True
    )

    fig3 = px.pie(
        counts, names="cluster", values="comments",
        hole=0.5,
        title="Audience Segment Distribution - Synthetic Dataset"
    )
    fig3.write_html(
        str(folder / "audience_segment_share.html"),
        include_plotlyjs=True
    )

    matrix_path = REPORT_DIR / "cluster_topic_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError("Run Phase 4 Part 3 first.")

    matrix = pd.read_csv(matrix_path, index_col=0)
    fig4 = px.imshow(
        matrix,
        labels={"x": "Synthetic Expected Topic",
                "y": "Discovered Cluster",
                "color": "Unique Texts"},
        title="Synthetic Topic vs Discovered Cluster",
        aspect="auto", text_auto=True
    )
    fig4.write_html(
        str(folder / "cluster_topic_heatmap.html"),
        include_plotlyjs=True
    )

    manifest = {
        "dataset": "Synthetic demonstration",
        "total_charts": 4,
        "charts": [
            "audience_cluster_sizes.html",
            "cluster_languages.html",
            "audience_segment_share.html",
            "cluster_topic_heatmap.html"
        ],
        "note": "Synthetic topic heatmap is an evaluation diagnostic."
    }
    (REPORT_DIR / "cluster_visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("INTERACTIVE CLUSTERING VISUALIZATIONS")
    for chart in manifest["charts"]:
        print("Created:", chart)
    print("PHASE 4 PART 4 COMPLETED")

if __name__ == "__main__":
    build_cluster_charts()
