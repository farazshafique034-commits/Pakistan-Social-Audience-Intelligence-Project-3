import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from project3.config.settings import PROCESSED_DIR, EXTERNAL_DIR, REPORT_DIR

def evaluate_clusters():
    comments = pd.read_csv(
        PROCESSED_DIR / "comments_interpreted.csv", encoding="utf-8-sig"
    )
    truth_path = EXTERNAL_DIR / "synthetic_comment_truth.csv"
    if not truth_path.exists():
        raise FileNotFoundError("Missing synthetic ground truth.")

    truth = pd.read_csv(truth_path, encoding="utf-8-sig")
    if truth["comment_id"].duplicated().any():
        raise ValueError("Duplicate ground-truth comment IDs.")

    data = comments.merge(
        truth, on="comment_id", how="left", validate="one_to_one"
    )

    if data["expected_topic"].isna().any():
        raise ValueError("Some comments have no expected topic.")

    if not data["data_origin"].eq("synthetic").all():
        raise ValueError("Synthetic labels cannot evaluate real comments.")

    valid = data[data["cluster_id"] >= 0].copy()
    unique = valid.drop_duplicates(subset=["clean_text"]).copy()

    if len(unique) < 3:
        raise ValueError("Insufficient unique texts for evaluation.")

    conflicts = valid.groupby("clean_text")["expected_topic"].nunique()
    conflicting_texts = int((conflicts > 1).sum())
    if conflicting_texts:
        raise ValueError("Identical texts have conflicting synthetic topics.")

    ari = adjusted_rand_score(
        unique["expected_topic"], unique["cluster_id"]
    )
    nmi = normalized_mutual_info_score(
        unique["expected_topic"], unique["cluster_id"]
    )

    matrix = pd.crosstab(
        unique["cluster_id"], unique["expected_topic"]
    )
    matrix.index.name = "cluster_id"

    purity = float(matrix.max(axis=1).sum() / matrix.to_numpy().sum())

    report = {
        "evaluation_type": "Synthetic ground-truth diagnostic",
        "evaluation_unit": "Unique cleaned text",
        "total_comments": int(len(data)),
        "evaluated_unique_texts": int(len(unique)),
        "number_of_clusters": int(unique["cluster_id"].nunique()),
        "number_of_expected_topics": int(unique["expected_topic"].nunique()),
        "adjusted_rand_index": round(float(ari), 4),
        "normalized_mutual_information": round(float(nmi), 4),
        "cluster_purity": round(purity, 4),
        "conflicting_identical_texts": conflicting_texts,
        "limitations": [
            "Synthetic comments are template-generated",
            "Metrics do not establish real-world performance",
            "The clustering model is unsupervised",
            "Unique training texts were used to select K",
            "This is a descriptive diagnostic, not held-out validation"
        ]
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(
        REPORT_DIR / "cluster_topic_matrix.csv", encoding="utf-8-sig"
    )
    (REPORT_DIR / "clustering_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("PROJECT 3 - CLUSTER EVALUATION")
    print("Unique texts:", len(unique))
    print("Adjusted Rand Index:", report["adjusted_rand_index"])
    print("Normalized Mutual Information:", report["normalized_mutual_information"])
    print("Cluster purity:", report["cluster_purity"])
    print("Synthetic diagnostics only; no real-world validation.")
    print("PHASE 4 PART 3 COMPLETED")

if __name__ == "__main__":
    evaluate_clusters()
