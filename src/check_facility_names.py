import pandas as pd

file_path = "data/raw/dataset1_ismr_aug2017_dec2018_facility_medicine.csv"

df = pd.read_csv(file_path)

print("Total rows:", len(df))
print("Missing facility names:", df["facility_name"].isna().sum())

facility_check = (
    df.groupby("facility_id")["facility_name"]
    .agg(
        total_rows="size",
        non_missing_names=lambda x: x.notna().sum(),
        unique_names=lambda x: x.dropna().unique().tolist()
    )
    .reset_index()
)

missing_facilities = facility_check[
    facility_check["non_missing_names"] == 0
]

print("\n===== FACILITIES WITH NO NAME ANYWHERE =====")
print("Count:", len(missing_facilities))

print("\nFirst 30:")
print(missing_facilities.head(30).to_string(index=False))

print("\n===== FACILITIES WITH BOTH MISSING AND NON-MISSING NAMES =====")

partial_facilities = facility_check[
    (facility_check["non_missing_names"] > 0) &
    (facility_check["non_missing_names"] < facility_check["total_rows"])
]

print("Count:", len(partial_facilities))

print(partial_facilities.head(30).to_string(index=False))