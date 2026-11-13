from project3.config.settings import RANDOM_SEED

TEXT_CONFIG = {
    "min_text_length": 2,
    "remove_duplicates": True,
    "preserve_roman_urdu": True,
    "preserve_urdu_script": True
}

VECTORIZER_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True
}

CLUSTERING_CONFIG = {
    "algorithm": "KMeans",
    "min_clusters": 3,
    "max_clusters": 12,
    "random_state": RANDOM_SEED,
    "n_init": 10
}

ANOMALY_CONFIG = {
    "algorithm": "IsolationForest",
    "contamination": 0.03,
    "n_estimators": 200,
    "random_state": RANDOM_SEED
}

ENGAGEMENT_FEATURES = [
    "likes",
    "comments_count",
    "shares",
    "views",
    "days_since_post"
]

PROJECT3_DASHBOARD_PAGES = [
    "Audience Clustering",
    "Content Intelligence",
    "Anomaly Detection",
    "Marketing Opportunities"
]
