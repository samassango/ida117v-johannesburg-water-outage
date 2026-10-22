"""Streamlit interface – run with:  streamlit run app.py"""
import streamlit as st, pandas as pd
from pipeline_utils import *

st.set_page_config(page_title="Johannesburg Water Outage Early Warning", page_icon="💧", layout="wide")
st.title("💧 Johannesburg Water Outage Early-Warning Tool")
st.caption("Predicts whether a ward is HIGH or LOW risk of a major outage (> 12 hours) in the next two-week period.")
with st.expander("Responsible-use notice (please read)", expanded=True):
    st.markdown("- Decision-support only: a human maintenance planner must review every flag.\n"
                "- Trained on **synthetic** ward-level data; recall on unseen periods was ~61%, so some high-risk wards will be missed.\n"
                "- Do **not** use to withhold services from any community, or for individual households.\n"
                "- No personal data is collected or stored.")

@st.cache_resource
def get_model(): return load_model()
model = get_model()
tab1, tab2 = st.tabs(["Single ward", "Upload CSV (batch)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    row = {}
    with c1:
        row["pipe_age_years"] = st.number_input("Pipe age (years)", 0.0, 100.0, 35.0, help="Average pipe age in the ward")
        row["pipe_material"] = st.selectbox("Main pipe material", MATERIALS)
        row["region"] = st.selectbox("Region", REGIONS)
    with c2:
        row["avg_pressure_kpa"] = st.number_input("Average pressure (kPa)", 0.0, 900.0, 360.0, help="Typical distribution-node pressure")
        row["reservoir_distance_km"] = st.number_input("Distance to reservoir (km)", 0.0, 30.0, 5.0)
        row["elevation_m"] = st.number_input("Elevation (m)", 1300.0, 2000.0, 1650.0)
    with c3:
        row["recent_complaints"] = st.number_input("Complaint tickets (last period)", 0, 50, 2)
        row["prior_outages_last_3_periods"] = st.number_input("Outages in previous 3 periods", 0, 3, 0)
        row["rainfall_mm"] = st.number_input("Rainfall (mm)", 0.0, 150.0, 10.0)
        row["max_temp_c"] = st.number_input("Max temperature (°C)", -5.0, 45.0, 22.0)
        row["informal_settlement_pct"] = st.number_input("Informal settlement (%)", 0.0, 100.0, 15.0)
    if st.button("Predict outage risk", type="primary"):
        try:
            label, prob, why = predict(model, [row])[0]
            (st.error if label == "High" else st.success)(f"Predicted class: **{label} risk**  |  Probability of high risk: **{prob:.0%}**")
            st.progress(min(prob, 1.0)); st.write(why)
        except ValueError as e:
            st.warning(f"Please fix the input: {e}")

with tab2:
    st.write("Upload a CSV with columns: " + ", ".join(list(LIMITS) + CAT))
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        try:
            d = pd.read_csv(f)
            missing = [c for c in list(LIMITS) + CAT if c not in d.columns]
            if missing: st.error(f"Missing columns: {', '.join(missing)}")
            else:
                out = predict(model, d[list(LIMITS) + CAT].to_dict("records"))
                d["predicted_risk"] = [o[0] for o in out]; d["probability_high"] = [round(o[1], 3) for o in out]
                st.dataframe(d); st.bar_chart(d.predicted_risk.value_counts())
                st.download_button("Download predictions", d.to_csv(index=False), "predictions.csv")
        except ValueError as e: st.warning(str(e))
        except Exception: st.error("Could not read this file. Check that it is a valid CSV.")
