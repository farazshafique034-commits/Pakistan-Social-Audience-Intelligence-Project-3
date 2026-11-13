import json
import numpy as np
import pandas as pd
import plotly.express as px

from project3.config.settings import PROCESSED_DIR, REPORT_DIR


def review_preprocessing():
    posts = pd.read_csv(PROCESSED_DIR / "posts_features.csv")
    comments = pd.read_csv(PROCESSED_DIR / "comments_features.csv")
    ml_data = pd.read_csv(PROCESSED_DIR / "post_ml_features.csv")

    charts_dir = REPORT_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    linked_ids = set(comments["post_id"])
    available_ids = set(posts["post_id"])
    unknown_ids = linked_ids - available_ids

    actual_counts = comments.groupby("post_id").size()
    observed = posts["post_id"].map(actual_counts).fillna(0).astype(int)
    expected = posts["comments_count"].astype(int)

    mismatches = int((observed != expected).sum())
    duplicate_posts = int(posts["post_id"].duplicated().sum())
    duplicate_comments = int(comments["comment_id"].duplicated().sum())
    empty_text = int(comments["clean_text"].fillna("").eq("").sum())

    numeric_matrix = ml_data.drop(columns=["post_id"]).to_numpy(dtype=float)
    nonfinite = int((~np.isfinite(numeric_matrix)).sum())

    passed = all([
        len(unknown_ids) == 0,
        mismatches == 0,
        duplicate_posts == 0,
        duplicate_comments == 0,
        empty_text == 0,
        nonfinite == 0,
        len(ml_data) == len(posts),
        ml_data["post_id"].equals(posts["post_id"])
    ])

    report = {
        "posts_checked": int(len(posts)),
        "comments_checked": int(len(comments)),
        "ml_features": int(len(ml_data.columns) - 1),
        "unknown_post_ids": int(len(unknown_ids)),
        "count_mismatches": mismatches,
        "duplicate_post_ids": duplicate_posts,
        "duplicate_comment_ids": duplicate_comments,
        "empty_cleaned_texts": empty_text,
        "nonfinite_ml_values": nonfinite,
        "repeated_comment_texts": int(
            comments["clean_text"].duplicated().sum()
        ),
        "data_origin": {
            str(k): int(v) for k, v in
            posts["data_origin"].value_counts().items()
        },
        "quality_checks_passed": bool(passed)
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "preprocessing_review.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    content = posts.groupby(
        ["content_type", "platform"], as_index=False
    )["likes_per_day"].median()

    fig1 = px.bar(
        content, x="content_type", y="likes_per_day",
        color="platform", barmode="group",
        title="Synthetic Content Performance: Median Likes Per Day",
        labels={"likes_per_day": "Median Likes Per Day"}
    )
    fig1.write_html(
        str(charts_dir / "content_performance.html"),
        include_plotlyjs=True
    )

    languages = comments["language_type"].value_counts().reset_index()
    languages.columns = ["language", "comments"]

    fig2 = px.bar(
        languages, x="language", y="comments",
        color="language",
        title="Synthetic Audience Language Distribution"
    )
    fig2.write_html(
        str(charts_dir / "language_distribution.html"),
        include_plotlyjs=True
    )

    platforms = posts["platform"].value_counts().reset_index()
    platforms.columns = ["platform", "posts"]

    fig3 = px.pie(
        platforms, names="platform", values="posts",
        title="Synthetic Social Media Platform Distribution"
    )
    fig3.write_html(
        str(charts_dir / "platform_distribution.html"),
        include_plotlyjs=True
    )

    print(json.dumps(report, indent=2))
    print("Interactive charts generated:", 3)

    if not passed:
        raise ValueError("Preprocessing validation failed.")

    print("PHASE 3 PART 4 COMPLETED")


if __name__ == "__main__":
    review_preprocessing()
