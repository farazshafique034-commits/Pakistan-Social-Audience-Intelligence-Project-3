import json
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)

from project3.config.settings import (
    PROCESSED_DIR, EXTERNAL_DIR, REPORT_DIR
)


def evaluate_anomalies():
    result_path = PROCESSED_DIR / "posts_with_anomalies.csv"
    truth_path = EXTERNAL_DIR / "synthetic_post_truth.csv"

    if not result_path.exists() or not truth_path.exists():
        raise FileNotFoundError("Missing anomaly results or synthetic truth.")

    posts = pd.read_csv(result_path, encoding="utf-8-sig")
    truth = pd.read_csv(truth_path, encoding="utf-8-sig")

    if posts["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs in anomaly results.")
    if truth["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs in synthetic ground truth.")

    data = posts.merge(
        truth, on="post_id", how="left", validate="one_to_one"
    )
    if data["expected_anomaly"].isna().any():
        raise ValueError("Missing synthetic ground-truth labels.")

    if not data["data_origin"].eq("synthetic").all():
        raise ValueError("Synthetic labels cannot evaluate real observations.")

    y_true = data["expected_anomaly"].ne("normal").astype(int)
    y_pred = data["is_anomaly"].astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    ap = float(average_precision_score(y_true, data["anomaly_score"]))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    k = max(1, int(y_true.sum()))
    top_k = data.nlargest(k, "anomaly_score")
    precision_at_k = float(
        top_k["expected_anomaly"].ne("normal").mean()
    )

    report = {
        "evaluation_dataset": "Synthetic planted anomalies",
        "total_posts": int(len(data)),
        "planted_anomalies": int(y_true.sum()),
        "detected_anomalies": int(y_pred.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "average_precision": round(ap, 4),
        "precision_at_k": round(precision_at_k, 4),
        "k": int(k),
        "limitations": [
            "Ground truth is synthetically planted.",
            "Scores do not establish real-world detection accuracy.",
            "Contamination sets an approximate anomaly fraction.",
            "Flagged posts require human investigation."
        ]
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chart_dir = REPORT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    (REPORT_DIR / "anomaly_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    data["ground_truth_class"] = data["expected_anomaly"].map(
        lambda value: "Normal" if value == "normal" else "Planted anomaly"
    )

    fig1 = px.histogram(
        data, x="anomaly_score", color="ground_truth_class",
        nbins=45, barmode="overlay", opacity=0.7,
        title="Isolation Forest Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score (Higher = More Unusual)"}
    )

    fig2 = px.scatter(
        data, x="views", y="likes", color="review_status",
        hover_data=["post_id", "platform", "content_type"],
        title="Post Engagement: Likes vs Views",
        labels={"views": "Views", "likes": "Likes"}
    )

    platform_counts = (
        data.groupby(["platform", "review_status"], as_index=False)
        .size().rename(columns={"size": "posts"})
    )
    fig3 = px.bar(
        platform_counts, x="platform", y="posts",
        color="review_status", barmode="group",
        title="Flagged Posts by Platform"
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual normal", "Actual anomaly"],
        columns=["Predicted normal", "Predicted anomaly"]
    )

    fig4 = px.imshow(
        matrix_df, text_auto=True, aspect="auto",
        title="Anomaly Detection Confusion Matrix",
        labels={"color": "Posts"}
    )

    charts = {
        "anomaly_score_distribution.html": fig1,
        "engagement_anomaly_scatter.html": fig2,
        "anomalies_by_platform.html": fig3,
        "anomaly_confusion_matrix.html": fig4
    }

    for filename, figure in charts.items():
        figure.write_html(
            str(chart_dir / filename), include_plotlyjs=True
        )

    # Complete standalone HTML summary with embedded charts
    parts = [
        "<html><head><meta charset=\"utf-8\">",
        "<title>Project 3 - Anomaly Analysis Report</title>",
        "</head><body style=\"font-family:Arial;margin:35px\">",
        "<h1>Pakistan Social Audience Intelligence</h1>",
        "<h2>Phase 5 - Anomaly Detection Report</h2>",
        "<p>Dataset: synthetic demonstration records.</p>",
        f"<p>Posts: {len(data)} | Planted anomalies: {int(y_true.sum())}</p>",
        f"<p>Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}</p>",
        "<p>Detected anomalies are investigation signals only.</p>"
    ]

    for figure in charts.values():
        parts.append(figure.to_html(
            full_html=False, include_plotlyjs="cdn"
        ))

    parts.append("</body></html>")
    (REPORT_DIR / "phase5_anomaly_report.html").write_text(
        "\n".join(parts), encoding="utf-8"
    )

    print("PROJECT 3 - PHASE 5 FINAL EVALUATION")
    print("Posts:", len(data))
    print("Planted anomalies:", int(y_true.sum()))
    print("Detected anomalies:", int(y_pred.sum()))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Average Precision:", round(ap, 4))
    print("Precision@K:", round(precision_at_k, 4))
    print("Interactive charts generated:", len(charts))
    print("PHASE 5 PART 3 COMPLETED")
    print("PHASE 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    evaluate_anomalies()
