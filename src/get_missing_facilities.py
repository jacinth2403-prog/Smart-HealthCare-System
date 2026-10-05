import pandas as pd

file_path = "data/raw/dataset1_ismr_aug2017_dec2018_facility_medicine.csv"

df = pd.read_csv(file_path)

missing = (
    df[df["facility_name"].isna()]
    [["facility_id"]]
    .drop_duplicates()
)

print("Missing facilities:", len(missing))
print()

print(missing.to_string(index=False))

missing.to_csv(
    "data/processed/missing_facilities.csv",
    index=False
)

print("\nSaved to:")
print("data/processed/missing_facilities.csv")