import io
import re
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def read_csv_bytes(blob, label):
    if len(blob) > 20 * 1024 * 1024:
        raise ValueError(f"{label} CSV exceeds the 20 MB limit.")
    try:
        df = pd.read_csv(
            io.BytesIO(blob), encoding="utf-8-sig",
            dtype={"post_id": "string", "comment_id": "string"},
            low_memory=False
        )
    except (UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read {label} CSV: {exc}") from exc
    df.columns = [str(x).strip().lower() for x in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"{label} contains duplicate column names.")
    if df.empty:
        raise ValueError(f"{label} CSV is empty.")
    return df


def clean_text(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"https?://\S+", " ", str(value), flags=re.I)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def analyze_upload(posts_bytes, comments_bytes):
    posts = read_csv_bytes(posts_bytes, "Posts")
    comments = read_csv_bytes(comments_bytes, "Comments")
    notes = []

    if len(posts) > 12000 or len(comments) > 25000:
        raise ValueError("Limit: 12,000 posts and 25,000 comments per upload.")

    required_posts = {"post_id", "platform", "likes"}
    required_comments = {"post_id", "comment_text"}
    missing_posts = required_posts - set(posts.columns)
    missing_comments = required_comments - set(comments.columns)
    if missing_posts:
        raise ValueError(f"Posts CSV missing: {sorted(missing_posts)}")
    if missing_comments:
        raise ValueError(f"Comments CSV missing: {sorted(missing_comments)}")

    for df in (posts, comments):
        df["post_id"] = df["post_id"].astype("string").str.strip()
        if df["post_id"].isna().any() or df["post_id"].eq("").any():
            raise ValueError("Missing or empty post_id detected.")

    if posts["post_id"].duplicated().any():
        raise ValueError("Posts CSV contains duplicate post_id values.")

    if "comment_id" not in comments.columns:
        comments.insert(
            0, "comment_id",
            [f"UPLOAD_{i:07d}" for i in range(len(comments))]
        )
        notes.append("Comment IDs generated for this upload.")

    if comments["comment_id"].isna().any():
        raise ValueError("Missing comment IDs.")
    if comments["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs.")

    comments["comment_text"] = (
        comments["comment_text"].fillna("").astype(str).str.strip()
    )
    if comments["comment_text"].eq("").any():
        raise ValueError("Comments CSV contains empty comment text.")

    unknown_ids = set(comments["post_id"]) - set(posts["post_id"])
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} comment post IDs do not exist in Posts CSV."
        )

    for column in ["brand", "campaign", "content_type"]:
        if column not in posts.columns:
            posts[column] = "Unknown"
        posts[column] = posts[column].fillna("Unknown").astype(str)

    posts["platform"] = posts["platform"].fillna("Unknown").astype(str)

    for df in (posts, comments):
        if "data_origin" not in df.columns:
            df["data_origin"] = "unspecified"
        df["data_origin"] = (
            df["data_origin"].fillna("unspecified").astype(str)
        )

    for column in ["likes", "views", "shares", "comments_count", "days_since_post"]:
        if column not in posts.columns:
            posts[column] = np.nan
        posts[column] = pd.to_numeric(posts[column], errors="coerce")
        if posts[column].dropna().lt(0).any():
            raise ValueError(f"Negative values in {column}.")

    if posts["likes"].isna().any():
        raise ValueError("All posts require valid numeric likes.")

    observed_counts = comments.groupby("post_id").size()
    observed = posts["post_id"].map(observed_counts).fillna(0)
    if posts["comments_count"].isna().any():
        posts["comments_count"] = posts["comments_count"].fillna(observed)
        notes.append("Missing comment counts derived from uploaded comments.")
    mismatches = int((posts["comments_count"] != observed).sum())
    if mismatches:
        notes.append(
            f"{mismatches} posts have public comment totals differing from uploaded rows."
        )

    if "language_type" not in comments.columns:
        comments["language_type"] = "Unspecified"
    comments["language_type"] = (
        comments["language_type"].fillna("Unspecified").astype(str)
    )

    if "clean_text" not in comments.columns:
        comments["clean_text"] = comments["comment_text"].map(clean_text)
    else:
        comments["clean_text"] = (
            comments["clean_text"].fillna("").astype(str).str.strip()
        )
        missing_clean = comments["clean_text"].eq("")
        comments.loc[missing_clean, "clean_text"] = (
            comments.loc[missing_clean, "comment_text"].map(clean_text)
        )

    previous_clusters = (
        "cluster_id" in comments.columns
        and comments["cluster_id"].notna().all()
    )

    if previous_clusters:
        comments["cluster_id"] = pd.factorize(
            comments["cluster_id"].astype(str)
        )[0]
        if "cluster_label" not in comments.columns:
            comments["cluster_label"] = (
                "Cluster " + comments["cluster_id"].astype(str)
            )
        else:
            comments["cluster_label"] = (
                comments["cluster_label"].fillna("Unlabeled").astype(str)
            )
        notes.append("Existing uploaded cluster assignments reused.")
    else:
        unique_texts = comments["clean_text"].dropna().unique().tolist()
        if len(unique_texts) < 4:
            comments["cluster_id"] = 0
            comments["cluster_label"] = "Insufficient text variety"
            notes.append("Too few distinct comments to train meaningful clusters.")
        else:
            training_texts = unique_texts[:10000]
            vectorizer = TfidfVectorizer(
                max_features=2500, ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b"
            )
            features = vectorizer.fit_transform(training_texts)
            k = min(8, max(2, int(np.sqrt(len(training_texts) / 3))))
            k = min(k, len(training_texts) - 1)
            model = MiniBatchKMeans(
                n_clusters=k, random_state=42,
                n_init=3, batch_size=512
            )
            model.fit(features)
            comments["cluster_id"] = model.predict(
                vectorizer.transform(comments["clean_text"].tolist())
            )
            vocabulary = vectorizer.get_feature_names_out()
            names = {}
            for cluster_id, center in enumerate(model.cluster_centers_):
                indices = np.argsort(center)[-3:][::-1]
                terms = [str(vocabulary[i]) for i in indices]
                names[cluster_id] = " / ".join(terms)
            comments["cluster_label"] = comments["cluster_id"].map(names)
            notes.append("New TF-IDF + MiniBatchKMeans clusters trained in memory.")

    existing_anomalies = False
    if "is_anomaly" in posts.columns:
        flags = pd.to_numeric(posts["is_anomaly"], errors="coerce")
        if flags.notna().all() and flags.isin([0, 1]).all():
            posts["is_anomaly"] = flags.astype(int)
            existing_anomalies = True

    if existing_anomalies:
        if "anomaly_score" not in posts.columns:
            posts["anomaly_score"] = np.nan
        notes.append("Existing uploaded anomaly flags reused.")
    elif len(posts) >= 30:
        views = posts["views"].replace(0, np.nan)
        base = pd.DataFrame({
            "log_likes": np.log1p(posts["likes"]),
            "log_comments": np.log1p(posts["comments_count"]),
            "log_views": np.log1p(posts["views"]),
            "log_shares": np.log1p(posts["shares"]),
            "likes_per_view": posts["likes"] / views,
            "missing_views": posts["views"].isna().astype(int),
            "missing_shares": posts["shares"].isna().astype(int)
        })
        base = base.replace([np.inf, -np.inf], np.nan)
        for column in base.columns:
            median = base[column].median()
            base[column] = base[column].fillna(
                float(median) if pd.notna(median) else 0.0
            )
        X = RobustScaler().fit_transform(base)
        detector = IsolationForest(
            n_estimators=150, contamination=0.03,
            random_state=42, n_jobs=-1
        )
        posts["is_anomaly"] = (detector.fit_predict(X) == -1).astype(int)
        posts["anomaly_score"] = -detector.score_samples(X)
        notes.append("New Isolation Forest fitted to uploaded posts in memory.")
    else:
        posts["is_anomaly"] = 0
        posts["anomaly_score"] = np.nan
        notes.append("Anomaly detection skipped: fewer than 30 posts.")

    posts["review_status"] = np.where(
        posts["is_anomaly"].eq(1), "Needs review", "Not flagged"
    )
    return posts, comments, notes
