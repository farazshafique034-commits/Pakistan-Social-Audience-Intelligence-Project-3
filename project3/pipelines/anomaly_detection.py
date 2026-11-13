import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from project3.config.settings import (
    PROCESSED_DIR, MODEL_DIR, REPORT_DIR, RANDOM_SEED
)
from project3.config.ml_config import ANOMALY_CONFIG


def detect_anomalies():
    source = PROCESSED_DIR / "anomaly_ml_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run anomaly feature preparation first.")

    ml = pd.read_csv(source, encoding="utf-8-sig")
    posts = pd.read_csv(
        PROCESSED_DIR / "posts_features.csv", encoding="utf-8-sig"
    )

    features = [c for c in ml.columns if c != "post_id"]
    if ml.empty or ml["post_id"].duplicated().any():
        raise ValueError("Invalid anomaly features.")

    X = ml[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Anomaly features contain invalid numbers.")

    model = IsolationForest(
        n_estimators=ANOMALY_CONFIG["n_estimators"],
        contamination=ANOMALY_CONFIG["contamination"],
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    predictions = model.fit_predict(X)
    scores = -model.score_samples(X)

    results = ml[["post_id"]].copy()
    results["anomaly_score"] = scores
    results["is_anomaly"] = (predictions == -1).astype(int)
    results["anomaly_rank"] = results["anomaly_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    output = posts.merge(
        results, on="post_id", how="left", validate="one_to_one"
    )
    if output["anomaly_score"].isna().any():
        raise ValueError("Some posts did not receive anomaly scores.")

    # Explanation is descriptive, not an Isolation Forest attribution.
    groups = ["platform", "content_type"]
    peer_likes = output.groupby(groups)["likes"].transform("median")
    peer_comments = output.groupby(groups)["comments_count"].transform("median")

    reasons = []
    for index, row in output.iterrows():
        if row["is_anomaly"] != 1:
            reasons.append("Normal model classification")
            continue

        notes = []
        if row["likes"] > 3 * max(float(peer_likes.loc[index]), 1):
            notes.append("Likes unusually high for comparable content")
        if row["likes"] < 0.2 * max(float(peer_likes.loc[index]), 1):
            notes.append("Likes unusually low for comparable content")
        if row["comments_count"] > 3 * max(float(peer_comments.loc[index]), 1):
            notes.append("Comment volume unusually high")
        if pd.isna(row["views"]):
            notes.append("View count unavailable")
        if pd.isna(row["shares"]):
            notes.append("Share count unavailable")
        if not notes:
            notes.append("Unusual multivariate engagement pattern")
        reasons.append("; ".join(notes))

    output["investigation_reason"] = reasons
    output["review_status"] = np.where(
        output["is_anomaly"] == 1, "Needs review", "Not flagged"
    )

    flagged = output[output["is_anomaly"] == 1].sort_values(
        "anomaly_score", ascending=False
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")

    output.to_csv(
        PROCESSED_DIR / "posts_with_anomalies.csv",
        index=False, encoding="utf-8-sig"
    )
    flagged.to_csv(
        REPORT_DIR / "detected_anomalies.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = {
        "algorithm": "IsolationForest",
        "total_posts": int(len(output)),
        "flagged_posts": int(len(flagged)),
        "flagged_percentage": round(100 * len(flagged) / len(output), 2),
        "configured_contamination": ANOMALY_CONFIG["contamination"],
        "feature_count": int(len(features)),
        "note": "Anomaly scores are investigation signals, not proof of fraud or errors.",
        "source": "Synthetic demonstration data"
    }

    (REPORT_DIR / "anomaly_detection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("PHASE 5 PART 2 COMPLETED")


if __name__ == "__main__":
    detect_anomalies()
