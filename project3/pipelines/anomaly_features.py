import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from project3.config.settings import PROCESSED_DIR, REPORT_DIR

FEATURES = [
    "log_likes",
    "log_comments",
    "log_shares",
    "log_views",
    "likes_per_day_log",
    "comments_per_day_log",
    "likes_per_view",
    "shares_per_view",
    "comments_per_view",
    "missing_views",
    "missing_shares",
    "peer_likes_deviation",
    "peer_comments_deviation"
]


def prepare_anomaly_features():
    source = PROCESSED_DIR / "posts_features.csv"
    posts = pd.read_csv(source, encoding="utf-8-sig")

    if posts.empty or posts["post_id"].duplicated().any():
        raise ValueError("Invalid or duplicate post records.")

    data = pd.DataFrame(index=posts.index)
    data["post_id"] = posts["post_id"].astype(str)

    likes = pd.to_numeric(posts["likes"], errors="coerce")
    comments = pd.to_numeric(posts["comments_count"], errors="coerce")
    shares = pd.to_numeric(posts["shares"], errors="coerce")
    views = pd.to_numeric(posts["views"], errors="coerce")
    days = pd.to_numeric(posts["days_since_post"], errors="coerce").clip(lower=1)

    data["log_likes"] = np.log1p(likes.clip(lower=0))
    data["log_comments"] = np.log1p(comments.clip(lower=0))
    data["log_shares"] = np.log1p(shares.clip(lower=0))
    data["log_views"] = np.log1p(views.clip(lower=0))
    data["likes_per_day_log"] = np.log1p((likes / days).clip(lower=0))
    data["comments_per_day_log"] = np.log1p((comments / days).clip(lower=0))

    valid_views = views.where(views > 0, np.nan)
    data["likes_per_view"] = likes / valid_views
    data["shares_per_view"] = shares / valid_views
    data["comments_per_view"] = comments / valid_views
    data["missing_views"] = views.isna().astype(int)
    data["missing_shares"] = shares.isna().astype(int)

    # Compare likes and comments against similar content.
    peer_columns = ["platform", "content_type"]
    peer_likes = posts.groupby(peer_columns)["likes"].transform("median")
    peer_comments = posts.groupby(peer_columns)["comments_count"].transform("median")

    data["peer_likes_deviation"] = np.log1p(likes) - np.log1p(peer_likes)
    data["peer_comments_deviation"] = (
        np.log1p(comments) - np.log1p(peer_comments)
    )

    matrix = data[FEATURES].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    fill_values = {}

    for column in FEATURES:
        median = matrix[column].median()
        replacement = float(median) if pd.notna(median) else 0.0
        fill_values[column] = replacement
        matrix[column] = matrix[column].fillna(replacement)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)

    if not np.isfinite(scaled).all():
        raise ValueError("Anomaly matrix contains non-finite values.")

    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    scaled_df.insert(0, "post_id", data["post_id"].values)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scaled_df.to_csv(
        PROCESSED_DIR / "anomaly_ml_features.csv",
        index=False, encoding="utf-8-sig"
    )

    report = {
        "total_posts": int(len(scaled_df)),
        "total_features": len(FEATURES),
        "feature_names": FEATURES,
        "missing_values_before_imputation": {
            str(k): int(v) for k, v in
            data[FEATURES].isna().sum().items()
        },
        "imputation_values": fill_values,
        "scaling": "RobustScaler",
        "peer_comparison": "platform + content_type",
        "note": "Synthetic demonstration dataset"
    }

    (REPORT_DIR / "anomaly_feature_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Posts prepared:", len(scaled_df))
    print("Anomaly ML features:", len(FEATURES))
    print("PHASE 5 PART 1 COMPLETED")


if __name__ == "__main__":
    prepare_anomaly_features()
