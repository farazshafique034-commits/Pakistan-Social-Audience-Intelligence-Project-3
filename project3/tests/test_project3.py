import unittest
import pandas as pd
import numpy as np

from project3.config.settings import (
    RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, REPORT_DIR
)
from project3.analytics.data_validation import (
    validate_posts, validate_comments
)


class TestProject3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.posts = pd.read_csv(
            RAW_DIR / "social_posts.csv", encoding="utf-8-sig"
        )
        cls.comments = pd.read_csv(
            RAW_DIR / "social_comments.csv", encoding="utf-8-sig"
        )

    def test_01_post_schema(self):
        result = validate_posts(self.posts)
        self.assertEqual(len(result), len(self.posts))

    def test_02_comment_schema(self):
        result = validate_comments(self.comments, self.posts)
        self.assertEqual(len(result), len(self.comments))

    def test_03_unique_post_ids(self):
        self.assertFalse(self.posts["post_id"].duplicated().any())

    def test_04_unique_comment_ids(self):
        self.assertFalse(self.comments["comment_id"].duplicated().any())

    def test_05_post_comment_links(self):
        unknown = set(self.comments["post_id"]) - set(self.posts["post_id"])
        self.assertEqual(len(unknown), 0)

    def test_06_post_comment_counts(self):
        counts = self.comments.groupby("post_id").size()
        observed = self.posts["post_id"].map(counts).fillna(0)
        expected = self.posts["comments_count"]
        self.assertTrue((observed == expected).all())

    def test_07_dataset_provenance(self):
        self.assertTrue(self.posts["data_origin"].eq("synthetic").all())
        self.assertTrue(self.comments["data_origin"].eq("synthetic").all())

    def test_08_cleaned_comments(self):
        data = pd.read_csv(PROCESSED_DIR / "comments_cleaned.csv")
        self.assertEqual(len(data), len(self.comments))
        self.assertFalse(data["clean_text"].fillna("").eq("").any())

    def test_09_engineered_features(self):
        data = pd.read_csv(PROCESSED_DIR / "posts_features.csv")
        self.assertEqual(len(data), len(self.posts))
        self.assertIn("engagement_total", data.columns)
        self.assertIn("likes_per_day", data.columns)

    def test_10_ml_feature_matrix(self):
        data = pd.read_csv(PROCESSED_DIR / "post_ml_features.csv")
        self.assertEqual(len(data), len(self.posts))
        matrix = data.drop(columns=["post_id"]).to_numpy(dtype=float)
        self.assertTrue(np.isfinite(matrix).all())

    def test_11_clustering_assignments(self):
        data = pd.read_csv(PROCESSED_DIR / "comments_interpreted.csv")
        self.assertEqual(len(data), len(self.comments))
        self.assertTrue(data["cluster_id"].notna().all())
        self.assertGreaterEqual(data["cluster_id"].nunique(), 2)

    def test_12_synthetic_topic_truth(self):
        truth = pd.read_csv(EXTERNAL_DIR / "synthetic_comment_truth.csv")
        self.assertEqual(len(truth), len(self.comments))
        self.assertFalse(truth["comment_id"].duplicated().any())

    def test_13_anomaly_output(self):
        data = pd.read_csv(PROCESSED_DIR / "posts_with_anomalies.csv")
        self.assertEqual(len(data), len(self.posts))
        self.assertTrue(data["is_anomaly"].isin([0, 1]).all())
        self.assertTrue(data["anomaly_score"].notna().all())

    def test_14_anomaly_truth(self):
        truth = pd.read_csv(EXTERNAL_DIR / "synthetic_post_truth.csv")
        self.assertEqual(len(truth), len(self.posts))

    def test_15_marketing_recommendations(self):
        data = pd.read_csv(REPORT_DIR / "marketing_recommendations.csv")
        self.assertGreater(len(data), 0)
        self.assertIn("recommended_action", data.columns)
        self.assertIn("evidence", data.columns)

    def test_16_marketing_content(self):
        data = pd.read_csv(REPORT_DIR / "marketing_content_performance.csv")
        self.assertGreater(len(data), 0)
        self.assertIn("median_likes_per_day", data.columns)

    def test_17_anomaly_evaluation(self):
        import json
        path = REPORT_DIR / "anomaly_evaluation.json"
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        for metric in ["precision", "recall", "f1_score"]:
            self.assertGreaterEqual(report[metric], 0)
            self.assertLessEqual(report[metric], 1)

    def test_18_clustering_evaluation(self):
        import json
        path = REPORT_DIR / "clustering_evaluation.json"
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        self.assertIn("adjusted_rand_index", report)
        self.assertIn("normalized_mutual_information", report)


if __name__ == "__main__":
    unittest.main()
