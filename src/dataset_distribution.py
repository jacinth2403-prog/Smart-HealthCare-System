import pandas as pd

df = pd.read_csv("data/processed/dataset1_clean.csv")

print("===== DATASET DISTRIBUTION =====")

print("\n--- Stockout rate ---")
print(df["stockout_flag"].value_counts(normalize=True) * 100)

print("\n--- Demand statistics ---")
print(df["demand_quantity"].describe())

print("\n--- Consumption statistics ---")
print(df["consumed_quantity"].describe())

print("\n--- Patient activity statistics ---")
print(df["patient_activity"].describe())

print("\n--- Facilities per month ---")
print(df.groupby("month")["facility_id"].nunique())

print("\n--- Medicines per month ---")
print(df.groupby("month")["medicine_id"].nunique())

print("\n--- Rows per month ---")
print(df["month"].value_counts().sort_index())

print("\n--- Stockout rate by month ---")
print(
    df.groupby("month")["stockout_flag"]
    .mean()
    .mul(100)
)