import pandas as pd
import streamlit as st
from project3.config.settings import PROCESSED_DIR, REPORT_DIR

@st.cache_data(show_spinner=False)
def load_project3_data():
    paths = {
        "posts": PROCESSED_DIR / "posts_with_anomalies.csv",
        "comments": PROCESSED_DIR / "comments_interpreted.csv",
        "segments": REPORT_DIR / "marketing_audience_needs.csv",
        "content": REPORT_DIR / "marketing_content_performance.csv",
        "campaigns": REPORT_DIR / "marketing_campaign_performance.csv",
        "recommendations": REPORT_DIR / "marketing_recommendations.csv"
    }
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Project 3 outputs: " + ", ".join(missing)
        )
    return {
        key: pd.read_csv(path, encoding="utf-8-sig")
        for key, path in paths.items()
    }


def project3_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)
    st.info(
        "Project 3 uses synthetic demonstration data. "
        "Charts are not verified observations of real brands."
    )


def load_or_stop():
    try:
        return load_project3_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.warning("Run the Project 3 Phase 4-6 pipelines first.")
        st.stop()


def csv_download(label, dataframe, filename):
    st.download_button(
        label,
        dataframe.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv"
    )
