import json
import numpy as np
import pandas as pd
import joblib
from project3.config.settings import PROCESSED_DIR, MODEL_DIR, REPORT_DIR

def interpret_clusters():
    source = PROCESSED_DIR / "comments_clustered.csv"
    if not source.exists():
        raise FileNotFoundError("Run Phase 4 Part 1 first.")

    comments = pd.read_csv(source, encoding="utf-8-sig")
    model = joblib.load(MODEL_DIR / "audience_kmeans.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    vocabulary = vectorizer.get_feature_names_out()

    if "cluster_id" not in comments.columns:
        raise ValueError("Missing cluster_id column.")

    rows = []
    label_map = {}

    for cluster_id in sorted(comments["cluster_id"].unique()):
        cid = int(cluster_id)
        subset = comments[comments["cluster_id"] == cid]
        if cid < 0:
            label = "Unassigned"
            terms = []
        else:
            center = model.cluster_centers_[cid]
            indices = np.argsort(center)[::-1][:8]
            terms = [str(vocabulary[i]) for i in indices]
            label = " / ".join(terms[:3])

        label_map[cid] = label
        samples = subset["comment_text"].dropna().astype(str).head(5).tolist()
        rows.append({
            "cluster_id": cid,
            "cluster_label": label,
            "comment_count": int(len(subset)),
            "percentage": round(100 * len(subset) / len(comments), 2),
            "top_keywords": ", ".join(terms),
            "sample_comments": " | ".join(samples),
            "unique_texts": int(subset["clean_text"].nunique())
        })

    comments["cluster_label"] = comments["cluster_id"].map(label_map)
    summary = pd.DataFrame(rows)

    comments.to_csv(
        PROCESSED_DIR / "comments_interpreted.csv",
        index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        REPORT_DIR / "audience_segments.csv",
        index=False, encoding="utf-8-sig"
    )

    report = {
        "algorithm": "K-Means",
        "labeling": "Automatic descriptive TF-IDF keywords",
        "verified_business_topics": False,
        "clusters": rows
    }
    (REPORT_DIR / "cluster_interpretation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(summary[["cluster_id", "cluster_label", "comment_count"]].to_string(index=False))
    print("PHASE 4 PART 2 COMPLETED")

if __name__ == "__main__":
    interpret_clusters()
