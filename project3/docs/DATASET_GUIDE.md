# Project 3 Dataset Architecture

## Dataset Tables
- Posts: one record per social media post.
- Comments: one record per comment, linked by post_id.

## Sources
- synthetic: generated demonstration records.
- public_api: permitted public API records.
- licensed_dataset: reusable research dataset.
- manual_import: manually supplied records.

## Quality Rules
- Post and comment IDs must be unique.
- Comments must reference valid post IDs.
- Engagement counts cannot be negative.
- Unknown shares or views may be missing.
- Synthetic and real records must remain distinguishable.
- Preserve source URLs and collection dates where available.
- Check platform terms and reuse permissions.

## Machine Learning Roadmap
1. Data generation or permitted collection
2. English and Roman Urdu cleaning
3. NLP features and audience clustering
4. Cluster evaluation and labeling
5. Engagement anomaly detection
6. Marketing recommendations
7. Existing Streamlit dashboard integration
