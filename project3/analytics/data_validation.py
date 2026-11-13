import pandas as pd

from project3.config.settings import (
    POST_COLUMNS, COMMENT_COLUMNS, DATA_ORIGINS
)


def validate_posts(posts: pd.DataFrame) -> pd.DataFrame:
    data = posts.copy()
    missing = [c for c in POST_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing post columns: {missing}")
    if data.empty:
        raise ValueError("Posts dataset is empty")
    if data["post_id"].isna().any():
        raise ValueError("Missing post IDs")
    if data["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs")

    for column in ["likes", "comments_count", "days_since_post"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any() or (data[column] < 0).any():
            raise ValueError(f"Invalid values in {column}")

    for column in ["shares", "views"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if (data[column].dropna() < 0).any():
            raise ValueError(f"Invalid values in {column}")

    data["data_origin"] = data["data_origin"].astype(str).str.strip().str.lower()
    if not data["data_origin"].isin(DATA_ORIGINS).all():
        raise ValueError("Invalid post data origin")

    return data


def validate_comments(comments: pd.DataFrame, posts=None) -> pd.DataFrame:
    data = comments.copy()
    missing = [c for c in COMMENT_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing comment columns: {missing}")
    if data.empty:
        raise ValueError("Comments dataset is empty")
    if data["comment_id"].isna().any():
        raise ValueError("Missing comment IDs")
    if data["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs")
    if data["post_id"].isna().any():
        raise ValueError("Missing post IDs in comments")

    empty_text = data["comment_text"].fillna("").astype(str).str.strip().eq("")
    if empty_text.any():
        raise ValueError("Empty comment text found")

    data["data_origin"] = data["data_origin"].astype(str).str.strip().str.lower()
    if not data["data_origin"].isin(DATA_ORIGINS).all():
        raise ValueError("Invalid comment data origin")

    if posts is not None:
        unknown = set(data["post_id"]) - set(posts["post_id"])
        if unknown:
            raise ValueError(f"{len(unknown)} unknown post IDs")

    return data


def dataset_summary(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    return {
        "total_posts": len(posts),
        "total_comments": len(comments),
        "brands": posts["brand"].nunique(),
        "platforms": posts["platform"].nunique(),
        "linked_posts": comments["post_id"].nunique(),
        "synthetic_posts": int(posts["data_origin"].eq("synthetic").sum()),
        "synthetic_comments": int(comments["data_origin"].eq("synthetic").sum())
    }
