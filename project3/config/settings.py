from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

PROJECT_NAME = "Pakistan Social Audience & Campaign Intelligence Engine"
PRODUCT_CATEGORY = "Womens Lawn Suits"
RANDOM_SEED = 42
TARGET_MIN_RECORDS = 2000
TARGET_MAX_RECORDS = 5000

BRANDS = ["Brand A", "Brand B", "Brand C"]
PLATFORMS = ["Instagram", "Facebook", "TikTok"]

CONTENT_TYPES = [
    "Reel", "Product Photo", "Influencer Post",
    "Styling Video", "Sale Announcement", "Eid Campaign"
]

CAMPAIGNS = [
    "11.11 Sale", "12.12 Sale", "Eid Collection",
    "Summer Lawn", "New Arrival"
]

LANGUAGES = ["English", "Roman Urdu", "Urdu", "Mixed"]

TOPICS = [
    "Price Inquiry", "Discount Inquiry",
    "Delivery Concern", "Product Availability",
    "Size and Fit", "Fabric Quality",
    "Purchase Interest", "General Feedback"
]

DATA_ORIGINS = [
    "synthetic", "public_api",
    "licensed_dataset", "manual_import"
]

POST_COLUMNS = [
    "post_id", "brand", "platform", "campaign",
    "content_type", "product_category", "likes",
    "comments_count", "shares", "views",
    "days_since_post", "source_url",
    "data_origin", "collected_at"
]

COMMENT_COLUMNS = [
    "comment_id", "post_id", "comment_text",
    "language_type", "source_url",
    "data_origin", "collected_at"
]

PROCESSED_COMMENT_COLUMNS = COMMENT_COLUMNS + [
    "clean_text", "topic", "sentiment",
    "purchase_intent", "cluster_id", "cluster_label"
]

for folder in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
