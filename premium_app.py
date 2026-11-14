import streamlit as st
from premium_core import inject_css

st.set_page_config(page_title="Social Intelligence Pro", page_icon="📊", layout="wide")
inject_css()

pages = {
    "PROJECT 3 - DATA & AI STUDIO": [
        st.Page("premium_pages/p3_csv_studio.py", title="CSV Upload & AI Studio", icon="📤"),
        st.Page("premium_pages/p3_social.py", title="Audience 3D Studio", icon="🧠")
    ],
    "PROJECT 3 - SOCIAL INTELLIGENCE": [
        st.Page("pages/8_Audience_Clustering.py", title="Audience Clustering", icon="👥"),
        st.Page("pages/9_Content_Intelligence.py", title="Content Intelligence", icon="📈"),
        st.Page("pages/10_Anomaly_Detection.py", title="Anomaly Detection", icon="🔎"),
        st.Page("pages/11_Marketing_Opportunities.py", title="Marketing Opportunities", icon="💡")
    ]
}

page = st.navigation(pages, position="sidebar", expanded=True)
page.run()
