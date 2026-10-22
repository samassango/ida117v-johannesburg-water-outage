"""Shared logic for the Johannesburg Water Outage Early-Warning tool."""
import joblib, pandas as pd, numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "water_outage_pipeline.joblib"
MATERIALS = ["PVC", "Cast iron", "Asbestos cement", "Steel"]
REGIONS = list("ABCDEFG")
# (min, max, label) – ranges taken from the cleaned training data
LIMITS = {
    "pipe_age_years": (0, 100, "Pipe age (years)"),
    "avg_pressure_kpa": (0, 900, "Average pressure (kPa)"),
    "reservoir_distance_km": (0, 30, "Distance to reservoir (km)"),
    "elevation_m": (1300, 2000, "Elevation (m)"),
    "recent_complaints": (0, 50, "Complaint tickets (last period)"),
    "rainfall_mm": (0, 150, "Rainfall (mm)"),
    "max_temp_c": (-5, 45, "Max temperature (°C)"),
    "informal_settlement_pct": (0, 100, "Informal settlement (%)"),
    "prior_outages_last_3_periods": (0, 3, "Outages in previous 3 periods"),
}
NUM = list(LIMITS) + ["age_x_lowpressure"]
CAT = ["pipe_material", "region"]
THRESHOLD = 0.5

def load_model():
    return joblib.load(MODEL_PATH)

def validate(row: dict):
    """Return a list of plain-language error messages (empty list = valid)."""
    errors = []
    for k, (lo, hi, label) in LIMITS.items():
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{label} is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi} (you entered {v}).")
    if row.get("pipe_material") not in MATERIALS:
        errors.append(f"Pipe material must be one of: {', '.join(MATERIALS)}.")
    if row.get("region") not in REGIONS:
        errors.append(f"Region must be one of: {', '.join(REGIONS)}.")
    return errors

def to_frame(rows):
    d = pd.DataFrame(rows)
    d["age_x_lowpressure"] = d.pipe_age_years / (d.avg_pressure_kpa.fillna(350).clip(lower=50) / 100)
    return d[NUM + CAT]

def explain(row, prob):
    """Brief, plain-language reasons based on the model's strongest drivers."""
    r = []
    if row["recent_complaints"] >= 5: r.append("a high number of recent complaint tickets")
    if row["avg_pressure_kpa"] < 300: r.append("low average water pressure")
    if row["prior_outages_last_3_periods"] >= 1: r.append("recent outage history")
    if row["pipe_material"] in ("Cast iron", "Asbestos cement"): r.append(f"{row['pipe_material'].lower()} pipes (higher failure rates in this data)")
    if row["pipe_age_years"] >= 45: r.append("old pipes")
    label = "HIGH risk" if prob >= THRESHOLD else "LOW risk"
    if r: return f"{label}. Main factors present: " + ", ".join(r) + "."
    return f"{label}. No strong risk factors were present in the values entered."

def predict(model, rows):
    """rows: list of dicts. Returns list of (label, probability, explanation) or raises ValueError."""
    for row in rows:
        errs = validate(row)
        if errs: raise ValueError(" ".join(errs))
    X = to_frame(rows)
    p = model.predict_proba(X)[:, 1]
    return [("High" if pi >= THRESHOLD else "Low", float(pi), explain(r, pi)) for r, pi in zip(rows, p)]
