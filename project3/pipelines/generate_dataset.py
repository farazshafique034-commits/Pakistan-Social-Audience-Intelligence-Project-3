import json
from pathlib import Path

import numpy as np
import pandas as pd

from project3.config.settings import (
    RAW_DIR, EXTERNAL_DIR, BASE_DIR,
    BRANDS, PLATFORMS, CAMPAIGNS, CONTENT_TYPES,
    PRODUCT_CATEGORY, RANDOM_SEED
)
from project3.analytics.data_validation import (
    validate_posts, validate_comments, dataset_summary
)

POST_COUNT = 2400
COMMENT_COUNT = 5000
CREATED_AT = "2026-11-13T00:00:00Z"

TOPIC_MESSAGES = {
    "Price Inquiry": {
        "English": ["Price please?", "What is the price?", "How much for this suit?"],
        "Roman Urdu": ["price kya hai?", "is suit ki qeemat bata dein", "kitne ka hai ye dress?"],
        "Urdu": ["اس سوٹ کی قیمت کیا ہے؟", "قیمت بتا دیں"]
    },
    "Discount Inquiry": {
        "English": ["Any discount available?", "Is there a sale offer?", "Do you have a promo code?"],
        "Roman Urdu": ["discount milega?", "sale kab lagegi?", "koi offer hai kya?"],
        "Urdu": ["کیا کوئی رعایت ہے؟", "سیل کب لگے گی؟"]
    },
    "Delivery Concern": {
        "English": ["When will my order arrive?", "Delivery is taking too long", "How many delivery days?"],
        "Roman Urdu": ["delivery kitne din mein hogi?", "mera parcel abhi tak nahi aya", "order kab milega?"],
        "Urdu": ["ڈیلیوری کتنے دن میں ہوگی؟", "میرا آرڈر کب آئے گا؟"]
    },
    "Product Availability": {
        "English": ["Is it available in Karachi?", "Is this design in stock?", "Can I order this item?"],
        "Roman Urdu": ["karachi mein available hai?", "ye design stock mein hai?", "is suit ki availability hai?"],
        "Urdu": ["کیا یہ دستیاب ہے؟", "کراچی میں مل سکتا ہے؟"]
    },
    "Size and Fit": {
        "English": ["Do you have medium size?", "Is the size chart available?", "Does it fit true to size?"],
        "Roman Urdu": ["medium size mil jayega?", "size chart bhej dein", "large size available hai?"],
        "Urdu": ["میڈیم سائز ہے؟", "سائز چارٹ بھیج دیں"]
    },
    "Fabric Quality": {
        "English": ["Is this pure lawn fabric?", "How is the fabric quality?", "Does the color fade?"],
        "Roman Urdu": ["kapra kaisa hai?", "fabric quality achi hai?", "rang to nahi utray ga?"],
        "Urdu": ["کپڑے کی کوالٹی کیسی ہے؟", "کیا یہ لان کا کپڑا ہے؟"]
    },
    "Purchase Interest": {
        "English": ["I want to order this", "Please send ordering details", "How can I buy it?"],
        "Roman Urdu": ["mujhe order karna hai", "inbox mein details bhej dein", "ye suit lena hai"],
        "Urdu": ["مجھے یہ خریدنا ہے", "آرڈر کی تفصیل بھیج دیں"]
    },
    "General Feedback": {
        "English": ["Beautiful collection!", "Love this design", "Very nice colors"],
        "Roman Urdu": ["bohat pyara design hai", "collection bohat achi hai", "colors zabardast hain"],
        "Urdu": ["بہت خوبصورت ڈیزائن ہے", "رنگ بہت اچھے ہیں"]
    }
}


def create_datasets():
    rng = np.random.default_rng(RANDOM_SEED)
    topics = list(TOPIC_MESSAGES)
    anomaly_ids = set(rng.choice(POST_COUNT, size=36, replace=False).tolist())

    weights = np.ones(POST_COUNT)
    for idx in anomaly_ids:
        weights[idx] = 12.0

    comments_per_post = (
        np.ones(POST_COUNT, dtype=int)
        + rng.multinomial(
            COMMENT_COUNT - POST_COUNT,
            weights / weights.sum()
        )
    )

    posts = []
    post_truth = []

    for idx in range(POST_COUNT):
        post_id = f"P{idx + 1:05d}"
        platform = str(rng.choice(PLATFORMS))
        content_type = str(rng.choice(CONTENT_TYPES))
        campaign = str(rng.choice(CAMPAIGNS))
        brand = str(rng.choice(BRANDS))

        content_multiplier = 1.7 if content_type in ("Reel", "Styling Video") else 1.0
        platform_multiplier = 1.5 if platform == "TikTok" else 1.0

        views_raw = max(
            200,
            int(rng.lognormal(9.2, 0.75) * content_multiplier * platform_multiplier)
        )
        likes = int(views_raw * rng.uniform(0.02, 0.12))
        shares = int(views_raw * rng.uniform(0.002, 0.025))
        anomaly_type = "normal"

        if idx in anomaly_ids:
            if idx % 2 == 0:
                anomaly_type = "engagement_spike"
                views_raw *= 7
                likes *= 9
                shares *= 8
            else:
                anomaly_type = "unusual_low_engagement"
                likes = int(likes * 0.03)
                shares = int(shares * 0.03)

        likes = min(likes, views_raw)
        shares = min(shares, views_raw)

        views = views_raw
        if content_type in ("Product Photo", "Sale Announcement") and rng.random() < 0.55:
            views = np.nan

        if rng.random() < 0.20:
            shares = np.nan

        posts.append({
            "post_id": post_id,
            "brand": brand,
            "platform": platform,
            "campaign": campaign,
            "content_type": content_type,
            "product_category": PRODUCT_CATEGORY,
            "likes": likes,
            "comments_count": int(comments_per_post[idx]),
            "shares": shares,
            "views": views,
            "days_since_post": int(rng.integers(1, 121)),
            "source_url": "",
            "data_origin": "synthetic",
            "collected_at": CREATED_AT
        })
        post_truth.append({
            "post_id": post_id,
            "expected_anomaly": anomaly_type
        })

    comments = []
    comment_truth = []
    comment_number = 0

    for idx, post in enumerate(posts):
        for _ in range(int(comments_per_post[idx])):
            topic = str(rng.choice(topics))
            language = str(rng.choice(
                ["English", "Roman Urdu", "Urdu", "Mixed"],
                p=[0.30, 0.47, 0.16, 0.07]
            ))

            choices = TOPIC_MESSAGES[topic]

            if language == "Mixed":
                english = str(rng.choice(choices["English"]))
                roman = str(rng.choice(choices["Roman Urdu"]))
                message = english + " " + roman
            else:
                message = str(rng.choice(choices[language]))

            comment_number += 1
            comment_id = f"C{comment_number:06d}"

            comments.append({
                "comment_id": comment_id,
                "post_id": post["post_id"],
                "comment_text": message,
                "language_type": language,
                "source_url": "",
                "data_origin": "synthetic",
                "collected_at": CREATED_AT
            })

            comment_truth.append({
                "comment_id": comment_id,
                "expected_topic": topic
            })

    posts_df = validate_posts(pd.DataFrame(posts))
    comments_df = validate_comments(pd.DataFrame(comments), posts_df)

    if int(posts_df["comments_count"].sum()) != len(comments_df):
        raise ValueError("Post-comment count mismatch")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    posts_df.to_csv(RAW_DIR / "social_posts.csv", index=False, encoding="utf-8-sig")
    comments_df.to_csv(RAW_DIR / "social_comments.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(post_truth).to_csv(
        EXTERNAL_DIR / "synthetic_post_truth.csv", index=False
    )
    pd.DataFrame(comment_truth).to_csv(
        EXTERNAL_DIR / "synthetic_comment_truth.csv", index=False
    )

    summary = dataset_summary(posts_df, comments_df)
    manifest = {
        "dataset_name": "Pakistan Fashion Social Intelligence Demo",
        "provenance": "100% synthetic - not real brand observations",
        "generated_at": CREATED_AT,
        "random_seed": RANDOM_SEED,
        "summary": summary,
        "files": {
            "posts": "project3/data/raw/social_posts.csv",
            "comments": "project3/data/raw/social_comments.csv",
            "post_truth": "project3/data/external/synthetic_post_truth.csv",
            "comment_truth": "project3/data/external/synthetic_comment_truth.csv"
        }
    }

    manifest_path = BASE_DIR / "config" / "data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return summary


if __name__ == "__main__":
    print("Generating Project 3 synthetic dataset...")
    result = create_datasets()
    print(json.dumps(result, indent=2))
    print("PROJECT 3 DATASET GENERATED SUCCESSFULLY")
