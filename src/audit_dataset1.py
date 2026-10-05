import pandas as pd
import os

file_path = "data/raw/dataset1_ismr_aug2017_dec2018_facility_medicine.csv"

df = pd.read_csv(file_path)

print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== COLUMNS =====")
for col in df.columns:
    print(col)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== UNIQUE VALUES =====")
print("Facilities:", df["facility_id"].nunique())
print("Medicines:", df["medicine_id"].nunique())
print("Months:", df["month"].nunique())

print("\n===== MONTHS =====")
print(df["month"].value_counts().sort_index())

print("\n===== DUPLICATES =====")
duplicates = df.duplicated(
    subset=["facility_id", "medicine_id", "month"]
).sum()

print("Duplicate facility-medicine-month rows:", duplicates)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== NUMERIC SUMMARY =====")
numeric_columns = [
    "patient_activity",
    "opening_stock",
    "received_quantity",
    "consumed_quantity",
    "closing_stock",
    "demand_quantity",
    "unfulfilled_demand",
    "stockout_flag",
    "reorder_level",
    "lead_time_days"
]

print(df[numeric_columns].describe())

print("\n===== NEGATIVE VALUES =====")

for col in numeric_columns:
    count = (df[col] < 0).sum()
    if count > 0:
        print(col, ":", count)

print("\n===== SAMPLE DATA =====")
print(df.head())

print("\n===== AUDIT COMPLETE =====")