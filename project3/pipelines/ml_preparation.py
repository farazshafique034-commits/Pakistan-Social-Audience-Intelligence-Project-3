import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from project3.config.settings import PROCESSED_DIR, MODEL_DIR

NUMERIC_COLUMNS = [
    "log_likes", "log_comments", "log_shares",
    "log_views", "likes_per_day",
    "comments_per_day", "engagement_rate_views",
    "days_since_post", "missing_views", "missing_shares"
]

CATEGORICAL_COLUMNS = [
    "platform", "content_type", "campaign"
]

def prepare_ml_features():
    source = PROCESSED_DIR / "posts_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run Phase 3 Part 2 first.")

    posts = pd.read_csv(source, encoding="utf-8-sig")
    if posts.empty or posts["post_id"].duplicated().any():
        raise ValueError("Invalid post dataset or duplicate IDs.")

    features = pd.DataFrame(index=posts.index)
    fill_values = {}

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(posts[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        median = values.median()
        replacement = float(median) if pd.notna(median) else 0.0
        fill_values[column] = replacement
        features[column] = values.fillna(replacement)

    for column in CATEGORICAL_COLUMNS:
        features[column] = posts[column].fillna("Unknown").astype(str)

    encoded = pd.get_dummies(
        features, columns=CATEGORICAL_COLUMNS, dtype=int
    )

    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(encoded)

    if not np.isfinite(scaled_values).all():
        raise ValueError("ML feature matrix contains invalid values.")

    output = pd.DataFrame(
        scaled_values, columns=encoded.columns
    )
    output.insert(0, "post_id", posts["post_id"].values)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        PROCESSED_DIR / "post_ml_features.csv",
        index=False, encoding="utf-8-sig"
    )

    joblib.dump({
        "scaler": scaler,
        "feature_columns": list(encoded.columns),
        "numeric_fill_values": fill_values,
        "categorical_columns": CATEGORICAL_COLUMNS
    }, MODEL_DIR / "post_feature_transformer.joblib")

    report = {
        "total_posts": int(len(output)),
        "total_ml_features": int(len(encoded.columns)),
        "missing_value_strategy": "Median or zero fallback",
        "scaling_algorithm": "RobustScaler",
        "feature_columns": list(encoded.columns)
    }

    (PROCESSED_DIR / "ml_preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Posts prepared:", len(output))
    print("ML features:", len(encoded.columns))
    print("Feature scaling: RobustScaler")
    print("PHASE 3 PART 3 COMPLETED")

if __name__ == "__main__":
    prepare_ml_features()
