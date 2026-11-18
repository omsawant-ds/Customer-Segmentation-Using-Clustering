"""Interactive customer-segmentation app.  Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.config import PALETTE  # noqa: E402
from segmentation.data import load_data  # noqa: E402
from segmentation.profile import persona_info  # noqa: E402
from segmentation.segmenter import RAW_COLUMNS, Segmenter  # noqa: E402

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Training segmentation model ...")
def get_model(k: int):
    return Segmenter(k=k).fit(load_data())


st.title("Credit-card customer segmentation")
st.caption("K-Means on log-scaled behavioural features, with each cluster mapped to a business persona.")

k = st.sidebar.slider("Number of segments (k)", 3, 8, 5)
if k != 5:
    st.sidebar.info("Persona names are tuned for k = 5; other values show generic names for extra segments.")
model = get_model(k)
colour = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(model.personas_.values()))}

overview, score, batch = st.tabs(["Segments overview", "Score one customer", "Score a CSV"])

with overview:
    prof = model.profile_.reset_index(drop=True)
    c1, c2 = st.columns([1, 1.4])
    c1.plotly_chart(px.bar(prof.sort_values("customers"), x="customers", y="persona", orientation="h",
                           color="persona", color_discrete_map=colour, text="share_%",
                           title="Segment sizes").update_layout(showlegend=False, yaxis_title=""),
                    use_container_width=True)
    proj = model.projection().sample(4000, random_state=0)
    c2.plotly_chart(px.scatter(proj, x="PC1", y="PC2", color="persona", color_discrete_map=colour, opacity=0.6,
                               title="Customers in PCA space").update_traces(marker_size=5),
                    use_container_width=True)
    st.subheader("Segment profiles (medians)")
    st.dataframe(prof, width="stretch", hide_index=True)
    st.subheader("What each segment means")
    for name in prof["persona"]:
        info = persona_info(name)
        st.markdown(f"**{name}**: {info['description']}  \n*Suggested action:* {info['action']}")

with score:
    st.write("Enter a customer's profile to see which segment they belong to.")
    a, b, c = st.columns(3)
    v = {}
    v["BALANCE"] = a.number_input("Balance", 0.0, 20000.0, 1000.0, 100.0)
    v["CREDIT_LIMIT"] = a.number_input("Credit limit", 0.0, 30000.0, 3000.0, 100.0)
    v["PURCHASES"] = a.number_input("Total purchases", 0.0, 50000.0, 500.0, 50.0)
    v["PURCHASES_TRX"] = a.number_input("Purchase transactions", 0, 300, 10)
    v["ONEOFF_PURCHASES"] = b.number_input("One-off purchases", 0.0, 50000.0, 200.0, 50.0)
    v["INSTALLMENTS_PURCHASES"] = b.number_input("Instalment purchases", 0.0, 50000.0, 100.0, 50.0)
    v["CASH_ADVANCE"] = b.number_input("Cash advance", 0.0, 50000.0, 0.0, 50.0)
    v["CASH_ADVANCE_TRX"] = b.number_input("Cash-advance transactions", 0, 100, 0)
    v["PAYMENTS"] = c.number_input("Payments", 0.0, 50000.0, 800.0, 50.0)
    v["MINIMUM_PAYMENTS"] = c.number_input("Minimum payments", 0.0, 50000.0, 200.0, 50.0)
    v["PRC_FULL_PAYMENT"] = c.slider("Share of full-payment months", 0.0, 1.0, 0.0)
    v["TENURE"] = c.slider("Tenure (months)", 6, 12, 12)
    v["BALANCE_FREQUENCY"] = a.slider("Balance update frequency", 0.0, 1.0, 1.0)
    v["PURCHASES_FREQUENCY"] = b.slider("Purchase frequency", 0.0, 1.0, 0.5)
    v["ONEOFF_PURCHASES_FREQUENCY"] = b.slider("One-off purchase frequency", 0.0, 1.0, 0.2)
    v["PURCHASES_INSTALLMENTS_FREQUENCY"] = b.slider("Instalment purchase frequency", 0.0, 1.0, 0.2)
    v["CASH_ADVANCE_FREQUENCY"] = c.slider("Cash-advance frequency", 0.0, 1.0, 0.0)
    result = model.predict(pd.DataFrame([v])[RAW_COLUMNS]).iloc[0]
    info = persona_info(result["persona"])
    st.success(f"### {result['persona']}")
    st.write(info["description"])
    st.info(f"**Suggested action:** {info['action']}")

with batch:
    st.write(f"Upload a CSV with the columns: `{', '.join(RAW_COLUMNS)}` (and optionally `CUST_ID`).")
    up = st.file_uploader("Customer CSV", type="csv")
    if up is not None:
        try:
            out = model.predict(pd.read_csv(up))
        except ValueError as e:
            st.error(str(e))
        else:
            st.dataframe(out.head(200), width="stretch", hide_index=True)
            st.bar_chart(out["persona"].value_counts())
            st.download_button("Download segments", out.to_csv(index=False), "segments.csv", "text/csv")
