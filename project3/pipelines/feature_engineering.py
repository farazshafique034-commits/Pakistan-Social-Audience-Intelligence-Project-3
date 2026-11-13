import json
import numpy as np
import pandas as pd
from project3.config.settings import PROCESSED_DIR

def build_features():
    posts_path = PROCESSED_DIR / "posts_verified.csv"
    comments_path = PROCESSED_DIR / "comments_cleaned.csv"

    if not posts_path.exists() or not comments_path.exists():
        raise FileNotFoundError("Run the previous Project 3 parts first.")

    posts = pd.read_csv(posts_path, encoding="utf-8-sig")
    comments = pd.read_csv(comments_path, encoding="utf-8-sig")

    posts["missing_shares"] = posts["shares"].isna().astype(int)
    posts["missing_views"] = posts["views"].isna().astype(int)

    post_comment_counts = comments.groupby("post_id").size()
    posts["observed_comments"] = (
        posts["post_id"].map(post_comment_counts).fillna(0).astype(int)
    )

    posts["engagement_total"] = (
        posts["likes"] + posts["comments_count"] + posts["shares"].fillna(0)
    )

    valid_views = posts["views"].where(posts["views"] > 0, np.nan)
    valid_days = posts["days_since_post"].clip(lower=1)

    posts["engagement_rate_views"] = posts["engagement_total"] / valid_views
    posts["likes_per_day"] = posts["likes"] / valid_days
    posts["comments_per_day"] = posts["comments_count"] / valid_days
    posts["comments_per_1000_views"] = (
        posts["comments_count"] / valid_views * 1000
    )

    posts["log_likes"] = np.log1p(posts["likes"])
    posts["log_comments"] = np.log1p(posts["comments_count"])
    posts["log_shares"] = np.log1p(posts["shares"].fillna(0))
    posts["log_views"] = np.log1p(posts["views"].fillna(0))

    comments["text_length"] = comments["clean_text"].fillna("").str.len()
    comments["word_count"] = (
        comments["clean_text"].fillna("").str.split().str.len()
    )
    comments["has_question"] = comments["comment_text"].fillna("").str.contains(
        r"[?؟]", regex=True
    ).astype(int)

    posts_output = PROCESSED_DIR / "posts_features.csv"
    comments_output = PROCESSED_DIR / "comments_features.csv"

    posts.to_csv(posts_output, index=False, encoding="utf-8-sig")
    comments.to_csv(comments_output, index=False, encoding="utf-8-sig")

    summary = {
        "posts": int(len(posts)),
        "comments": int(len(comments)),
        "mean_engagement": round(float(posts["engagement_total"].mean()), 2),
        "average_likes_per_day": round(float(posts["likes_per_day"].mean()), 2),
        "posts_with_missing_views": int(posts["missing_views"].sum()),
        "posts_with_missing_shares": int(posts["missing_shares"].sum()),
        "available_engagement_rates": int(posts["engagement_rate_views"].notna().sum()),
        "feature_files": [posts_output.name, comments_output.name]
    }

    (PROCESSED_DIR / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("PHASE 3 PART 2 COMPLETED")
    return posts, comments

if __name__ == "__main__":
    build_features()
