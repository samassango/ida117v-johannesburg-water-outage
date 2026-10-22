def explain_prediction(row):

    reasons = []

    if row["recent_complaints"] >= 4:
        reasons.append("Many complaints")

    if row["prior_outages_last_3_periods"] > 0:
        reasons.append("Recent outage history")

    if row["pipe_age_years"] > 40:
        reasons.append("Old pipes")

    if row["avg_pressure_kpa"] < 250:
        reasons.append("Low pressure")

    return reasons