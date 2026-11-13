import json
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from project3.config.settings import (
    PROCESSED_DIR, MODEL_DIR, REPORT_DIR, RANDOM_SEED
)
from project3.config.ml_config import (
    VECTORIZER_CONFIG, CLUSTERING_CONFIG
)


def run_clustering():
    source = PROCESSED_DIR / "comments_features.csv"

    if not source.exists():
        raise FileNotFoundError("Run Phase 3 preprocessing first.")

    comments = pd.read_csv(source, encoding="utf-8-sig")
    texts = comments["clean_text"].fillna("").astype(str).str.strip()
    valid = texts.ne("")

    if valid.sum() == 0:
        raise ValueError("No usable comments found.")

    unique_texts = sorted(set(texts[valid].tolist()))

    if len(unique_texts) < 5:
        raise ValueError("Not enough unique comments for clustering.")

    vectorizer = TfidfVectorizer(
        **VECTORIZER_CONFIG,
        token_pattern=r"(?u)\b\w+\b"
    )

    X = vectorizer.fit_transform(unique_texts)

    min_k = max(2, CLUSTERING_CONFIG["min_clusters"])
    max_k = min(
        8, CLUSTERING_CONFIG["max_clusters"], len(unique_texts) - 1
    )

    if min_k > max_k:
        raise ValueError("Insufficient unique texts for cluster search.")

    best_model = None
    best_k = None
    best_score = float("-inf")
    scores = []

    print("Starting TF-IDF + K-Means clustering...")
    print("Unique training texts:", len(unique_texts))
    print("TF-IDF vocabulary size:", X.shape[1])

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10
        )
        labels = model.fit_predict(X)
        discovered = len(np.unique(labels))

        if discovered < 2 or discovered >= X.shape[0]:
            continue

        score = float(silhouette_score(
            X, labels, metric="cosine"
        ))

        scores.append({
            "k": int(k),
            "silhouette_cosine": round(score, 4),
            "discovered_clusters": int(discovered)
        })

        print(f"K={k} | Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    if best_model is None:
        raise ValueError("No valid clustering model found.")

    comments["cluster_id"] = -1
    comments.loc[valid, "cluster_id"] = best_model.predict(
        vectorizer.transform(texts[valid].tolist())
    )

    terms = vectorizer.get_feature_names_out()
    cluster_rows = []

    for cluster_id in range(best_k):
        group = comments[comments["cluster_id"] == cluster_id]
        center = best_model.cluster_centers_[cluster_id]
        top_indices = np.argsort(center)[-8:][::-1]
        top_terms = ", ".join(terms[top_indices])

        cluster_rows.append({
            "cluster_id": int(cluster_id),
            "comments": int(len(group)),
            "percentage": round(100 * len(group) / len(comments), 2),
            "top_terms": top_terms,
            "example_comment": (
                str(group["comment_text"].iloc[0])
                if len(group) else ""
            )
        })

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib"
    )
    joblib.dump(
        best_model, MODEL_DIR / "audience_kmeans.joblib"
    )

    comments.to_csv(
        PROCESSED_DIR / "comments_clustered.csv",
        index=False, encoding="utf-8-sig"
    )

    summary = pd.DataFrame(cluster_rows)
    summary.to_csv(
        REPORT_DIR / "audience_cluster_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    diagnostics = {
        "algorithm": "TF-IDF + K-Means",
        "training_unit": "Unique cleaned comment text",
        "dataset_origin_note": "Synthetic demonstration data",
        "total_comments": int(len(comments)),
        "unique_training_texts": int(len(unique_texts)),
        "vocabulary_size": int(X.shape[1]),
        "selected_clusters": int(best_k),
        "best_silhouette_cosine": round(best_score, 4),
        "candidate_scores": scores
    }

    (REPORT_DIR / "clustering_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8"
    )

    print("")
    print("Selected number of clusters:", best_k)
    print("Best cosine silhouette score:", round(best_score, 4))
    print(summary.to_string(index=False))
    print("PHASE 4 PART 1 COMPLETED")


if __name__ == "__main__":
    run_clustering()
