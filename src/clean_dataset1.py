import pandas as pd

input_file = "data/raw/dataset1_ismr_aug2017_dec2018_facility_medicine.csv"
output_file = "data/processed/dataset1_clean.csv"

df = pd.read_csv(input_file)

print("Original rows:", len(df))
print("Original facilities:", df["facility_id"].nunique())

# Remove facilities with missing facility names
df_clean = df.dropna(subset=["facility_name"]).copy()

print("\nAfter removing unnamed facilities:")
print("Rows:", len(df_clean))
print("Facilities:", df_clean["facility_id"].nunique())
print("Removed rows:", len(df) - len(df_clean))
print("Removed facilities:", df["facility_id"].nunique() - df_clean["facility_id"].nunique())

# Save cleaned dataset
df_clean.to_csv(output_file, index=False)

print("\nSaved cleaned dataset to:")
print(output_file)