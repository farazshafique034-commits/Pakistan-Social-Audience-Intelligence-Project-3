import json
import pandas as pd
from project3.config.settings import PROCESSED_DIR, REPORT_DIR


def load_data():
    comments_path = PROCESSED_DIR / "comments_interpreted.csv"
    posts_path = PROCESSED_DIR / "posts_with_anomalies.csv"
    if not comments_path.exists() or not posts_path.exists():
        raise FileNotFoundError("Complete Phases 4 and 5 first.")
    comments = pd.read_csv(comments_path, encoding="utf-8-sig")
    posts = pd.read_csv(posts_path, encoding="utf-8-sig")
    if comments["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment IDs.")
    if posts["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs.")
    return comments, posts


def audience_intelligence(comments, posts):
    columns = ["post_id", "platform", "campaign", "content_type", "brand"]
    linked = comments.merge(
        posts[columns], on="post_id", how="left", validate="many_to_one"
    )
    if linked["platform"].isna().any():
        raise ValueError("Unlinked comments found.")

    grouped = linked.groupby(
        ["cluster_id", "cluster_label"], as_index=False, dropna=False
    ).agg(
        total_comments=("comment_id", "count"),
        unique_posts=("post_id", "nunique"),
        sample_comment=("comment_text", "first")
    )
    grouped["audience_share_pct"] = (
        grouped["total_comments"] / len(linked) * 100
    ).round(2)
    grouped = grouped.sort_values(
        "total_comments", ascending=False
    )

    platform_segments = linked.groupby(
        ["platform", "cluster_id"], as_index=False
    ).size().rename(columns={"size": "comments"})

    language_segments = linked.groupby(
        ["cluster_id", "language_type"], as_index=False
    ).size().rename(columns={"size": "comments"})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(
        REPORT_DIR / "marketing_audience_needs.csv",
        index=False, encoding="utf-8-sig"
    )
    platform_segments.to_csv(
        REPORT_DIR / "marketing_platform_segments.csv",
        index=False, encoding="utf-8-sig"
    )
    language_segments.to_csv(
        REPORT_DIR / "marketing_language_segments.csv",
        index=False, encoding="utf-8-sig"
    )

    print("PHASE 6 PART 1 COMPLETED")
    return grouped, linked


def content_intelligence(posts):
    df = posts.copy()
    df["days_since_post"] = pd.to_numeric(
        df["days_since_post"], errors="coerce"
    ).clip(lower=1)
    for col in ["likes", "comments_count", "shares", "views"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["likes_per_day"] = df["likes"] / df["days_since_post"]
    df["comments_per_day"] = (
        df["comments_count"] / df["days_since_post"]
    )
    df["views_available"] = df["views"].notna().astype(int)
    df["shares_available"] = df["shares"].notna().astype(int)

    summary = df.groupby(
        ["platform", "content_type"], as_index=False
    ).agg(
        total_posts=("post_id", "count"),
        median_likes_per_day=("likes_per_day", "median"),
        median_comments_per_day=("comments_per_day", "median"),
        median_views=("views", "median"),
        view_coverage=("views_available", "mean"),
        share_coverage=("shares_available", "mean"),
        flagged_posts=("is_anomaly", "sum")
    )
    for col in ["view_coverage", "share_coverage"]:
        summary[col] = (summary[col] * 100).round(1)

    summary.to_csv(
        REPORT_DIR / "marketing_content_performance.csv",
        index=False, encoding="utf-8-sig"
    )

    campaign = df.groupby(
        ["platform", "campaign"], as_index=False
    ).agg(
        total_posts=("post_id", "count"),
        median_likes_per_day=("likes_per_day", "median"),
        median_comments_per_day=("comments_per_day", "median"),
        flagged_posts=("is_anomaly", "sum")
    )
    campaign.to_csv(
        REPORT_DIR / "marketing_campaign_performance.csv",
        index=False, encoding="utf-8-sig"
    )

    print("PHASE 6 PART 2 COMPLETED")
    return summary


def make_recommendations(audience, content, posts):
    recommendations = []

    for _, row in audience.iterrows():
        count = int(row["total_comments"])
        label = str(row["cluster_label"])
        if int(row["cluster_id"]) < 0 or count < 10:
            continue
        recommendations.append({
            "priority": "Review",
            "category": "Audience needs",
            "finding": f"Cluster {int(row["cluster_id"])}: {label}",
            "evidence": f"{count} comments across {int(row["unique_posts"])} posts",
            "recommended_action": (
                "Review representative comments, confirm the audience need, "
                "then test relevant FAQs, creatives or product information."
            ),
            "validation": "Human review of cluster meaning",
            "data_basis": "Synthetic demonstration"
        })

    eligible = content[content["total_posts"] >= 10]
    if not eligible.empty:
        best = eligible.sort_values(
            "median_likes_per_day", ascending=False
        ).iloc[0]
        recommendations.append({
            "priority": "Test",
            "category": "Content strategy",
            "finding": (
                "{} on {} ".format(best["content_type"], best["platform"]) +
                "has the highest observed median likes/day among eligible groups."
            ),
            "evidence": (
                "{} posts; ".format(int(best["total_posts"])) +
                "median {:.1f} likes/day".format(best["median_likes_per_day"])
            ),
            "recommended_action": (
                "Test this content format against alternatives using "
                "comparable posting windows and campaign objectives."
            ),
            "validation": "Controlled campaign comparison",
            "data_basis": "Synthetic demonstration"
        })

    flagged = int(posts["is_anomaly"].sum())
    if flagged > 0:
        recommendations.append({
            "priority": "Investigate",
            "category": "Unusual engagement",
            "finding": f"{flagged} posts flagged by Isolation Forest",
            "evidence": "Model anomaly flags and engagement features",
            "recommended_action": (
                "Inspect flagged posts manually, check measurement "
                "availability, posting age and campaign context."
            ),
            "validation": "Manual anomaly investigation",
            "data_basis": "Synthetic demonstration"
        })

    result = pd.DataFrame(recommendations)
    result.to_csv(
        REPORT_DIR / "marketing_recommendations.csv",
        index=False, encoding="utf-8-sig"
    )

    report = {
        "project": "Pakistan Social Audience Intelligence",
        "dataset_type": "Synthetic demonstration",
        "total_recommendations": len(result),
        "limitations": [
            "Cluster names are preliminary keyword descriptions.",
            "Synthetic engagement is not actual brand performance.",
            "No purchase conversion or advertising ROAS is inferred.",
            "Suggested actions require testing before adoption."
        ],
        "recommendations": result.to_dict(orient="records")
    }
    (REPORT_DIR / "marketing_intelligence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("PHASE 6 PART 3 COMPLETED")
    print("Total recommendations:", len(result))


def main():
    comments, posts = load_data()
    audience, linked = audience_intelligence(comments, posts)
    content = content_intelligence(posts)
    make_recommendations(audience, content, posts)
    print("PHASE 6 COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()

