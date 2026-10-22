import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("water_outage_pipeline.joblib")

pipeline = load_pipeline()

st.set_page_config(
    page_title="Johannesburg Water Outage Early Warning",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Johannesburg Water Outage Early Warning System")

st.warning(
    """
    Responsible Use Notice
    
    • Decision support only.
    • Model trained on synthetic demonstration data.
    • Predictions should not be used to withhold services.
    • No personal information is stored.
    """
)

# --------------------------------------------------
# Validation function
# --------------------------------------------------

def validate_inputs(age, pressure, material):

    errors = []

    if age < 0 or age > 100:
        errors.append(
            "Pipe age must be between 0 and 100 years."
        )

    if pressure < 0 or pressure > 900:
        errors.append(
            "Average pressure must be between 0 and 900 kPa."
        )

    valid_materials = [
        "PVC",
        "Cast iron",
        "Asbestos cement",
        "Steel"
    ]

    if material not in valid_materials:
        errors.append(
            "Pipe material must be one of: PVC, Cast iron, Asbestos cement, Steel."
        )

    return errors


# --------------------------------------------------
# Single Prediction
# --------------------------------------------------

st.header("Single Ward Prediction")

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox(
        "Region",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    pipe_material = st.selectbox(
        "Pipe Material",
        ["PVC", "Cast iron", "Asbestos cement", "Steel"]
    )

    pipe_age = st.number_input(
        "Pipe Age (Years)",
        min_value=0,
        max_value=100,
        value=20
    )

    avg_pressure = st.number_input(
        "Average Pressure (kPa)",
        min_value=0,
        max_value=900,
        value=350
    )

with col2:
    reservoir_distance = st.number_input(
        "Reservoir Distance (km)",
        min_value=0.0,
        max_value=50.0,
        value=5.0
    )

    elevation = st.number_input(
        "Elevation (m)",
        min_value=0.0,
        max_value=2500.0,
        value=1600.0
    )

    complaints = st.number_input(
        "Recent Complaints",
        min_value=0,
        max_value=50,
        value=1
    )

    prior_outages = st.number_input(
        "Prior Outages (Last 3 Periods)",
        min_value=0,
        max_value=3,
        value=0
    )

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

temperature = st.number_input(
    "Maximum Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0
)

informal_share = st.number_input(
    "Informal Settlement Share (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

if st.button("Predict Outage Risk"):

    errors = validate_inputs(
        pipe_age,
        avg_pressure,
        pipe_material
    )

    if errors:
        for e in errors:
            st.error(e)

    else:

        data = pd.DataFrame([{
            "region": region,
            "pipe_material": pipe_material,
            "pipe_age_years": pipe_age,
            "avg_pressure_kpa": avg_pressure,
            "reservoir_distance_km": reservoir_distance,
            "elevation_m": elevation,
            "recent_complaints": complaints,
            "rainfall_mm": rainfall,
            "max_temp_c": temperature,
            "informal_settlement_pct": informal_share,
            "prior_outages_last_3_periods": prior_outages
        }])

        prediction = pipeline.predict(data)[0]

        probability = float(
            pipeline.predict_proba(data)[0][1]
        )

        if prediction == 1:

            st.error(
                f"⚠️ HIGH RISK ({probability:.1%})"
            )

            reasons = []

            if complaints >= 4:
                reasons.append("many complaints")

            if avg_pressure < 250:
                reasons.append("low pressure")

            if prior_outages > 0:
                reasons.append("recent outage history")

            if pipe_age > 40:
                reasons.append("old pipes")

            if pipe_material == "Cast iron":
                reasons.append("cast iron infrastructure")

            st.write("Likely drivers:")
            st.write(", ".join(reasons))

        else:

            st.success(
                f"✅ LOW RISK ({probability:.1%})"
            )

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

st.header("Batch CSV Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    predictions = pipeline.predict(df)

    probabilities = pipeline.predict_proba(df)[:, 1]

    df["Prediction"] = [
        "High Risk" if p == 1 else "Low Risk"
        for p in predictions
    ]

    df["Probability"] = probabilities

    st.dataframe(df)

    st.bar_chart(df["Probability"])

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )