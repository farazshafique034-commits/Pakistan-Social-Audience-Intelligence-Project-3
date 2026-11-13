import json
from datetime import datetime, timezone

import pandas as pd

from project3.config.settings import RAW_DIR, REPORT_DIR
from project3.analytics.data_validation import (
    validate_posts, validate_comments, dataset_summary
)


def inspect_dataset():
    posts_path = RAW_DIR / "social_posts.csv"
    comments_path = RAW_DIR / "social_comments.csv"

    for path in [posts_path, comments_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

    posts = validate_posts(pd.read_csv(posts_path))
    comments = validate_comments(pd.read_csv(comments_path), posts)

    actual_counts = comments.groupby("post_id").size()
    actual = posts["post_id"].map(actual_counts).fillna(0).astype(int)
    expected = posts["comments_count"].astype(int)

    mismatches = int((actual != expected).sum())

    report = {
        "project": "Pakistan Social Audience Intelligence",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": dataset_summary(posts, comments),
        "platform_distribution": posts["platform"].value_counts().to_dict(),
        "content_distribution": posts["content_type"].value_counts().to_dict(),
        "campaign_distribution": posts["campaign"].value_counts().to_dict(),
        "language_distribution": comments["language_type"].value_counts().to_dict(),
        "post_missing_values": posts.isna().sum().astype(int).to_dict(),
        "comment_missing_values": comments.isna().sum().astype(int).to_dict(),
        "repeated_comment_text_rows": int(
            comments.duplicated(subset=["post_id", "comment_text"]).sum()
        ),
        "missing_views": int(posts["views"].isna().sum()),
        "missing_shares": int(posts["shares"].isna().sum()),
        "post_comment_count_mismatches": mismatches,
        "post_data_origins": posts["data_origin"].value_counts().to_dict(),
        "comment_data_origins": comments["data_origin"].value_counts().to_dict(),
        "quality_checks_passed": mismatches == 0
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "data_quality_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=" * 55)
    print("PROJECT 3 - DATA QUALITY REPORT")
    print("=" * 55)
    print("Posts:", len(posts))
    print("Comments:", len(comments))
    print("Platforms:", posts["platform"].nunique())
    print("Content types:", posts["content_type"].nunique())
    print("Missing views:", report["missing_views"])
    print("Missing shares:", report["missing_shares"])
    print("Post/comment mismatches:", mismatches)
    print("Report saved:", output)

    if mismatches:
        raise ValueError("Dataset quality check failed")

    print("PHASE 2 PART 2 COMPLETED SUCCESSFULLY")
    return report


if __name__ == "__main__":
    inspect_dataset()
