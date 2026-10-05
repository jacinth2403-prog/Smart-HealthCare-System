import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, atan2

MASTER_FILE = "data/processed/master_dataset.csv"
DEMAND_MODEL_FILE = "src/demand_model.pkl"
OUTPUT_FILE = "data/processed/final_redistribution_recommendations.csv"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


print("Loading master dataset...")

df = pd.read_csv(MASTER_FILE)

latest_month = df["month"].max()
df = df[df["month"] == latest_month].copy()

print("Latest month:", latest_month)


# --------------------------------------------------
# 1. LOAD DEMAND MODEL
# --------------------------------------------------

print("Loading demand model...")

model_data = joblib.load(DEMAND_MODEL_FILE)

demand_model = model_data["model"]
preprocessor = model_data["preprocessor"]
features = model_data["features"]


# --------------------------------------------------
# 2. PREDICT DEMAND
# --------------------------------------------------

print("Predicting demand...")

X = df[features]
X_processed = preprocessor.transform(X)

df["predicted_demand"] = demand_model.predict(X_processed)
df["predicted_demand"] = df["predicted_demand"].clip(lower=0)


# --------------------------------------------------
# 3. IDENTIFY NEED AND SURPLUS USING REORDER LEVEL
# --------------------------------------------------

df["shortage"] = (
    df["reorder_level"] - df["closing_stock"]
).clip(lower=0)

df["surplus"] = (
    df["closing_stock"] - df["reorder_level"]
).clip(lower=0)


# --------------------------------------------------
# 4. PRIORITY SCORE
# --------------------------------------------------

# Higher predicted demand relative to available stock
df["demand_pressure"] = (
    df["predicted_demand"] /
    (df["closing_stock"] + 1)
)

# Higher shortage means greater need
df["shortage_pressure"] = (
    df["shortage"] /
    (df["reorder_level"] + 1)
)

# Combine the two signals
df["priority_score"] = (
    df["shortage_pressure"] * 0.6
    + df["demand_pressure"] * 0.4
)


# --------------------------------------------------
# 5. SHORTAGE / SURPLUS FACILITIES
# --------------------------------------------------

shortages = df[df["shortage"] > 0].copy()

surpluses = df[df["surplus"] > 0].copy()

print("Shortage rows:", len(shortages))
print("Surplus rows:", len(surpluses))


# --------------------------------------------------
# 6. ONLY USE RECORDS WITH USABLE COORDINATES
# --------------------------------------------------

shortages = shortages[
    shortages["latitude"].notna()
    & shortages["longitude"].notna()
    & (shortages["coordinate_usable"] == True)
].copy()

surpluses = surpluses[
    surpluses["latitude"].notna()
    & surpluses["longitude"].notna()
    & (surpluses["coordinate_usable"] == True)
].copy()

print("Shortages with coordinates:", len(shortages))
print("Surpluses with coordinates:", len(surpluses))


# --------------------------------------------------
# 7. PRIORITIZE HIGH-NEED FACILITIES
# --------------------------------------------------

shortages = shortages.sort_values(
    "priority_score",
    ascending=False
)


# --------------------------------------------------
# 8. REDISTRIBUTION
# --------------------------------------------------

recommendations = []

MAX_RECOMMENDATIONS_PER_SHORTAGE = 2


for shortage_index, shortage in shortages.iterrows():

    medicine_id = shortage["medicine_id"]
    remaining_need = shortage["shortage"]

    if remaining_need <= 0:
        continue

    donors = surpluses[
        surpluses["medicine_id"] == medicine_id
    ].copy()

    if len(donors) == 0:
        continue

    donors["distance_km"] = donors.apply(
        lambda row: haversine(
            shortage["latitude"],
            shortage["longitude"],
            row["latitude"],
            row["longitude"]
        ),
        axis=1
    )

    donors = donors.sort_values("distance_km")

    recommendation_count = 0

    for donor_index, donor in donors.iterrows():

        if remaining_need <= 0:
            break

        if recommendation_count >= MAX_RECOMMENDATIONS_PER_SHORTAGE:
            break

        available_surplus = donor["surplus"]

        if available_surplus <= 0:
            continue

        transfer_quantity = min(
            remaining_need,
            available_surplus
        )

        recommendations.append({
            "medicine_id": medicine_id,
            "medicine_name": shortage["medicine_name"],

            "from_facility_id": donor["facility_id"],
            "from_facility_name": donor["facility_name"],

            "to_facility_id": shortage["facility_id"],
            "to_facility_name": shortage["facility_name"],

            "from_district": donor["district"],
            "to_district": shortage["district"],

            "transfer_quantity": int(transfer_quantity),

            "distance_km": round(
                donor["distance_km"],
                2
            ),

            "predicted_demand": round(
                shortage["predicted_demand"],
                2
            ),

            "current_stock": shortage["closing_stock"],

            "reorder_level": shortage["reorder_level"],

            "shortage": round(
                shortage["shortage"],
                2
            ),

            "priority_score": round(
                shortage["priority_score"],
                4
            ),

            "stockout_flag": int(
                shortage["stockout_flag"]
            ),

            "from_latitude": donor["latitude"],
            "from_longitude": donor["longitude"],

            "to_latitude": shortage["latitude"],
            "to_longitude": shortage["longitude"]
        })

        remaining_need -= transfer_quantity

        surpluses.loc[
            donor_index,
            "surplus"
        ] -= transfer_quantity

        recommendation_count += 1


# --------------------------------------------------
# 9. SAVE
# --------------------------------------------------

recommendations_df = pd.DataFrame(recommendations)

print("\n===== FINAL REDISTRIBUTION ENGINE =====")

print("Latest month:", latest_month)
print("Shortage rows:", len(shortages))
print("Surplus rows:", len(surpluses))
print("Recommendations:", len(recommendations_df))


if len(recommendations_df) > 0:

    recommendations_df = recommendations_df.sort_values(
        by=["priority_score", "distance_km"],
        ascending=[False, True]
    )

    recommendations_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nTop recommendations:")

    print(
        recommendations_df[
            [
                "medicine_name",
                "from_facility_name",
                "to_facility_name",
                "transfer_quantity",
                "distance_km",
                "predicted_demand",
                "current_stock",
                "reorder_level",
                "shortage",
                "priority_score"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nSaved to:", OUTPUT_FILE)

else:

    print("\nNo recommendations generated.")