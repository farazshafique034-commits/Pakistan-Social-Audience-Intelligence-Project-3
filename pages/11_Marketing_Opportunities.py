import streamlit as st
import plotly.express as px
from project3.analytics.dashboard_data import (
    load_or_stop, project3_header, csv_download
)

project3_header(
    "Marketing Opportunities & Recommendations",
    "Project 3 | Evidence-led marketing decisions"
)

data = load_or_stop()
ideas = data["recommendations"].copy()
audience = data["segments"].copy()
content = data["content"].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Recommendations", len(ideas))
c2.metric("Audience Segments", len(audience))
c3.metric("Content Groups", len(content))

if ideas.empty:
    st.warning("No recommendations generated.")
    st.stop()

categories = sorted(ideas["category"].dropna().unique())
selected = st.multiselect(
    "Filter recommendation categories",
    categories, default=categories
)
filtered = ideas[ideas["category"].isin(selected)]

st.subheader("Actionable Marketing Recommendations")
for _, item in filtered.iterrows():
    heading = str(item["category"]) + ": " + str(item["finding"])
    with st.expander(heading):
        st.markdown("**Evidence:** " + str(item["evidence"]))
        st.markdown(
            "**Recommended action:** " + str(item["recommended_action"])
        )
        st.markdown("**Validation:** " + str(item["validation"]))

st.subheader("Audience Opportunity Distribution")
fig = px.bar(
    audience, x="cluster_id", y="total_comments",
    color="cluster_label",
    title="Comment Volume by Discovered Audience Group"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Content Opportunities")
eligible = content[content["total_posts"] >= 10]
if not eligible.empty:
    fig2 = px.bar(
        eligible, x="content_type", y="median_likes_per_day",
        color="platform", barmode="group",
        title="Synthetic Content-Format Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Not enough comparable posts for a content chart.")

st.subheader("Recommendation Table")
st.dataframe(filtered, use_container_width=True, hide_index=True)
csv_download(
    "Download marketing recommendations",
    filtered, "project3_marketing_recommendations.csv"
)

st.warning(
    "These recommendations are exploratory and based on synthetic data. "
    "Confirm segment meanings and test interventions before real use."
)
