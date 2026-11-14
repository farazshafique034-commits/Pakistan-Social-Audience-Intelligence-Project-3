from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent

def inject_css():
    st.markdown("""
<style>
.stApp {background:linear-gradient(145deg,#07111f,#0d1b2c 55%,#081220);color:#eaf3ff;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#0a182c,#08101d);border-right:1px solid #223650;}
section[data-testid="stSidebar"] a {border-radius:10px;}
section[data-testid="stSidebar"] a:hover {background:#173354;}
.block-container {padding-top:2rem;max-width:1550px;}
div[data-testid="stMetric"] {background:#102137;border:1px solid #28415f;padding:16px;border-radius:16px;}
div[data-testid="stMetricValue"] {color:#f7fbff;}
div[data-testid="stPlotlyChart"] {background:#0b192b;border:1px solid #233c57;border-radius:16px;padding:8px;}
.premium-eyebrow {color:#65d9ff;font-size:13px;font-weight:700;letter-spacing:2px;margin-bottom:7px;}
.premium-hero {font-size:38px;font-weight:800;color:#f0f7ff;margin-bottom:8px;}
.premium-sub {color:#afc3d8;font-size:15px;margin-bottom:22px;}
.premium-brand {padding:12px;background:#10243c;border:1px solid #345675;border-radius:15px;text-align:center;}
</style>
""", unsafe_allow_html=True)

def heading(tag, title, description):
    st.markdown(
        f"<div class=\"premium-eyebrow\">{tag}</div>"
        f"<div class=\"premium-hero\">{title}</div>"
        f"<div class=\"premium-sub\">{description}</div>",
        unsafe_allow_html=True
    )

def plot(fig, height=480):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b192b",
        plot_bgcolor="#0e2338",
        font=dict(color="#e3effc"),
        height=height,
        margin=dict(l=24, r=24, t=65, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_csv(required):
    folder = ROOT / "data"
    if not folder.exists():
        return None, None
    for path in sorted(folder.rglob("*.csv")):
        try:
            if path.stat().st_size > 30000000:
                continue
            columns = set(
                str(c).strip().lower()
                for c in pd.read_csv(path, nrows=0).columns
            )
            if required.issubset(columns):
                df = pd.read_csv(path)
                df.columns = [str(c).strip().lower() for c in df.columns]
                return df, str(path.relative_to(ROOT))
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
    return None, None

def demo_sentiment():
    rng = np.random.default_rng(42)
    messages = [
        "Beautiful collection and quality",
        "Price bohat zyada hai",
        "Delivery kitne din mein hogi",
        "Good design and beautiful colors",
        "I want to order this",
        "Price please and discount?",
        "I love these dresses",
        "Fabric quality could be better"
    ]
    n = 900
    return pd.DataFrame({
        "comment_text": rng.choice(messages, n),
        "sentiment": rng.choice(
            ["Positive", "Neutral", "Negative"], n,
            p=[0.44, 0.31, 0.25]
        ),
        "video_title": rng.choice(
            ["Fashion Haul", "Product Review",
             "Shopping Experience", "Customer Discussion"], n
        )
    })

def demo_ads():
    rng = np.random.default_rng(42)
    n = 1800
    impressions = rng.integers(3000, 80000, n)
    clicks = (impressions * rng.uniform(0.01, 0.07, n)).astype(int)
    conversions = (clicks * rng.uniform(0.015, 0.09, n)).astype(int)
    spend = rng.uniform(1500, 25000, n).round(2)
    revenue = (conversions * rng.uniform(1800, 9500, n)).round(2)
    dates = np.array(pd.date_range("2026-07-01", periods=120))
    return pd.DataFrame({
        "platform": rng.choice(
            ["Facebook", "Instagram", "Google", "TikTok", "YouTube"], n
        ),
        "campaign": rng.choice(
            ["Summer Sale", "11.11", "Eid", "Retargeting", "Launch"], n
        ),
        "date": rng.choice(dates, n),
        "spend": spend,
        "revenue": revenue,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions
    })

def text_3d(data, text_col, label_col, title):
    df = data.dropna(subset=[text_col]).copy()
    df = df[df[text_col].astype(str).str.strip().ne("")]
    if len(df) < 5:
        return None
    df = df.sample(n=min(700, len(df)), random_state=42).copy()
    vectorizer = TfidfVectorizer(
        max_features=1500, ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"
    )
    matrix = vectorizer.fit_transform(df[text_col].astype(str))
    if min(matrix.shape) <= 3:
        return None
    xyz = TruncatedSVD(
        n_components=3, random_state=42
    ).fit_transform(matrix)
    df["Axis 1"] = xyz[:, 0]
    df["Axis 2"] = xyz[:, 1]
    df["Axis 3"] = xyz[:, 2]
    df[label_col] = df[label_col].fillna("Unknown").astype(str)
    fig = px.scatter_3d(
        df, x="Axis 1", y="Axis 2", z="Axis 3",
        color=label_col, hover_data=[text_col],
        opacity=0.82, title=title
    )
    fig.update_traces(marker=dict(size=5))
    return fig

def sankey(data, left, right, value=None, title="Audience Flow"):
    if value is None:
        grouped = data.groupby(
            [left, right], as_index=False
        ).size().rename(columns={"size": "_flow"})
        value = "_flow"
    else:
        grouped = data.groupby(
            [left, right], as_index=False
        )[value].sum()
    grouped = grouped.dropna(subset=[left, right, value])
    grouped = grouped[grouped[value] > 0]
    left_names = sorted(grouped[left].astype(str).unique())
    right_names = sorted(grouped[right].astype(str).unique())
    labels = left_names + right_names
    left_ids = {name: i for i, name in enumerate(left_names)}
    right_ids = {
        name: i + len(left_names)
        for i, name in enumerate(right_names)
    }
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=16, label=labels
        ),
        link=dict(
            source=[left_ids[str(x)] for x in grouped[left]],
            target=[right_ids[str(x)] for x in grouped[right]],
            value=grouped[value].tolist()
        )
    ))
    fig.update_layout(title=title)
    return fig

