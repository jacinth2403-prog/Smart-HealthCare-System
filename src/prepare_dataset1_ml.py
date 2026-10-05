import pandas as pd

input_file = "data/processed/dataset1_clean.csv"

demand_output = "data/processed/dataset1_demand_ml.csv"
stockout_output = "data/processed/dataset1_stockout_ml.csv"

df = pd.read_csv(input_file)

features = [
    "facility_id",
    "medicine_id",
    "medicine_unit",
    "month",
    "patient_activity",
    "opening_stock",
    "reorder_level",
    "lead_time_days"
]

demand_df = df[features + ["demand_quantity"]].copy()

stockout_df = df[features + ["stockout_flag"]].copy()

demand_df.to_csv(demand_output, index=False)
stockout_df.to_csv(stockout_output, index=False)

print("Dataset 1 ML preparation completed.")
print()
print("Demand dataset:")
print(demand_df.shape)
print(demand_df.columns.tolist())
print()
print("Stockout dataset:")
print(stockout_df.shape)
print(stockout_df.columns.tolist())