# Pakistan Social Audience & Campaign Intelligence Engine

## Big Data Analytics - Project 3

**Student:** Faraz Shafique
**Student ID:** F2023376197
**Section:** A4
**University:** University of Management and Technology (UMT), Lahore

## Overview

This project explores audience behaviour and campaign
opportunities using text clustering, engagement analysis,
anomaly detection and interactive visualization.

Project 3 is implemented inside the existing AdSpend IQ
Streamlit application while retaining separate ML modules.

## Main Objectives

1. Discover audience groups from social-media comments.
2. Compare content formats and marketing campaigns.
3. Identify unusually performing social-media posts.
4. Translate findings into recommendations for marketers.
5. Present insights through an interactive dashboard.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF and K-Means
- Isolation Forest
- Streamlit and Plotly
- Joblib

## Data

The current demonstration uses synthetically generated
Pakistani fashion campaign records, not scraped or
verified measurements from real brands.

Default generator creates 2,400 social posts and 5,000
linked audience comments across simulated Instagram,
Facebook and TikTok content.

Source labels and synthetic ground truth are retained.

## Machine Learning Pipeline

1. Generate synthetic posts and comments.
2. Validate records and post-comment relationships.
3. Clean English, Roman Urdu and Urdu comments.
4. Engineer engagement and text features.
5. Train and evaluate TF-IDF + K-Means clustering.
6. Train Isolation Forest to flag unusual engagement.
7. Generate marketing intelligence and recommendations.

## Reproduce the Pipeline

From the ADSpend_IQ project root, activate the venv
and run the following Python modules in order:

    python -m project3.pipelines.generate_dataset
    python -m project3.analytics.data_quality
    python -m project3.pipelines.prepare_dataset
    python -m project3.pipelines.text_cleaning
    python -m project3.pipelines.feature_engineering
    python -m project3.pipelines.ml_preparation
    python -m project3.analytics.preprocessing_review
    python -m project3.pipelines.audience_clustering
    python -m project3.analytics.cluster_interpretation
    python -m project3.analytics.cluster_evaluation
    python -m project3.analytics.cluster_visualization
    python -m project3.pipelines.anomaly_features
    python -m project3.pipelines.anomaly_detection
    python -m project3.analytics.anomaly_evaluation
    python -m project3.analytics.marketing_intelligence

## Streamlit Dashboard

Start the existing application:

    python -m streamlit run app.py

Project 3 dashboard pages:

- Audience Clustering
- Content Intelligence
- Anomaly Detection
- Marketing Opportunities

## Testing

Run:

    python -m unittest project3.tests.test_project3 -v
    python -m project3.tests.final_audit

## Limitations

- Synthetic data does not measure real Pakistani brands.
- Clusters are exploratory and require interpretation.
- Model validation on synthetic data is not evidence
  of performance on unseen real-world campaigns.
- Anomaly flags are not proof of manipulation or fraud.
- Engagement signals cannot establish advertising ROAS.
- Real datasets require permitted access and provenance.

## Project Links

Project 2 - AdSpend IQ:
https://github.com/farazshafique034-commits/ADSpend_IQ-Project-2

Project 1 - SHEIN Sentiment Analysis:
https://github.com/farazshafique034-commits/SHEIN-REVIEWS-SENTIMENT-ANALYSIS

## Future Integration

Project 1 sentiment analysis results and Project 2
advertising analytics can be connected through explicit
import and export interfaces. Such connections must be
validated before describing them as implemented.
