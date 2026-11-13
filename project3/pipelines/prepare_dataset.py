import json
import pandas as pd
from project3.config.settings import RAW_DIR, PROCESSED_DIR
from project3.analytics.data_validation import validate_posts, validate_comments

def prepare_dataset():
    posts_file = RAW_DIR / "social_posts.csv"
    comments_file = RAW_DIR / "social_comments.csv"

    if not posts_file.exists() or not comments_file.exists():
        raise FileNotFoundError("Run the Project 3 dataset generator first.")

    posts = pd.read_csv(posts_file, encoding="utf-8-sig", dtype={"post_id": "string"})
    comments = pd.read_csv(comments_file, encoding="utf-8-sig", dtype={"post_id": "string"})

    posts = validate_posts(posts)
    comments = validate_comments(comments, posts)

    posts["post_id"] = posts["post_id"].str.strip()
    comments["post_id"] = comments["post_id"].str.strip()
    comments["comment_text"] = comments["comment_text"].astype("string").str.strip()

    if posts["post_id"].duplicated().any():
        raise ValueError("Duplicate normalized post IDs")
    if posts["post_id"].eq("").any() or comments["post_id"].eq("").any():
        raise ValueError("Empty normalized post ID")

    unknown = set(comments["post_id"]) - set(posts["post_id"])
    if unknown:
        raise ValueError(f"Unlinked comments: {len(unknown)} post IDs")

    columns = ["post_id", "brand", "platform", "campaign", "content_type"]
    context = comments.merge(
        posts[columns], on="post_id", how="left", validate="many_to_one"
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    posts.to_csv(PROCESSED_DIR / "posts_verified.csv", index=False, encoding="utf-8-sig")
    comments.to_csv(PROCESSED_DIR / "comments_verified.csv", index=False, encoding="utf-8-sig")
    context.to_csv(PROCESSED_DIR / "comments_context.csv", index=False, encoding="utf-8-sig")

    actual = comments.groupby("post_id").size()
    observed = posts["post_id"].map(actual).fillna(0).astype(int)
    declared = posts["comments_count"].astype(int)

    summary = {
        "posts": int(len(posts)),
        "comments": int(len(comments)),
        "linked_comments": int(len(context)),
        "count_mismatches": int((observed != declared).sum()),
        "platforms": posts["platform"].value_counts().to_dict(),
        "data_origins": comments["data_origin"].value_counts().to_dict()
    }

    (PROCESSED_DIR / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("PHASE 2 PART 4 COMPLETED")
    return posts, comments

if __name__ == "__main__":
    prepare_dataset()
