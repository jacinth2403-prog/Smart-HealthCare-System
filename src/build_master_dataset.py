import pandas as pd

dataset1_file = "data/processed/dataset1_clean.csv"
dataset2_file = "data/raw/Dataset_2_Disease_Outbreaks_TamilNadu_Prototype.xlsx"
dataset3_file = "data/raw/tamil_nadu_relevant_district_demographics.xlsx"
dataset4_file = "data/raw/tamil_nadu_ismr_facility_coordinates.xlsx"

output_file = "data/processed/master_dataset.csv"

print("Loading Dataset 1...")
df1 = pd.read_csv(dataset1_file)

print("Loading Dataset 2...")
df2 = pd.read_excel(
    dataset2_file,
    sheet_name="District_Month_Signal"
)

print("Loading Dataset 3...")
df3 = pd.read_excel(
    dataset3_file,
    sheet_name="District Demographics",
    header=4
)

crosswalk = pd.read_excel(
    dataset3_file,
    sheet_name="HUD Crosswalk",
    header=4
)

print("Loading Dataset 4...")
df4 = pd.read_excel(
    dataset4_file,
    sheet_name="Facility Coordinates",
    header=4
)

# --------------------------------------------------
# DATASET 4: Facility -> HUD -> District
# --------------------------------------------------

facility = df4[
    [
        "facility_id",
        "facility_name",
        "ISMR_HUD",
        "ISMR_block",
        "ISMR_facility_type",
        "latitude",
        "longitude",
        "match_confidence",
        "coordinate_match_status"
    ]
].copy()

facility = facility.merge(
    crosswalk[
        [
            "Source HUD name",
            "Mapped 2011 Census district"
        ]
    ],
    left_on="ISMR_HUD",
    right_on="Source HUD name",
    how="left"
)

facility = facility.drop(columns=["Source HUD name"])

facility = facility.rename(
    columns={
        "Mapped 2011 Census district": "district"
    }
)

# --------------------------------------------------
# NORMALIZE DISTRICT NAMES
# --------------------------------------------------

def normalize_district(value):
    if pd.isna(value):
        return value

    value = str(value).strip().upper()

    aliases = {
        "ARIYALLUR": "ARIYALUR"
    }

    return aliases.get(value, value)

facility["district_normalized"] = (
    facility["district"].apply(normalize_district)
)

df2["district_normalized"] = (
    df2["district_normalized"].apply(normalize_district)
)

df3["district_normalized"] = (
    df3["District (2011 Census)"].apply(normalize_district)
)

# --------------------------------------------------
# DATASET 3: Population
# --------------------------------------------------

population = df3[
    [
        "district_normalized",
        "Total population",
        "Male population",
        "Male (%)",
        "Female population",
        "Female (%)"
    ]
].copy()

population = population.rename(
    columns={
        "Total population": "population",
        "Male population": "male_population",
        "Male (%)": "male_percentage",
        "Female population": "female_population",
        "Female (%)": "female_percentage"
    }
)

# --------------------------------------------------
# DATASET 2: Disease / Outbreak Signals
# --------------------------------------------------

outbreak = df2[
    [
        "district_normalized",
        "report_month",
        "outbreak_count",
        "total_cases",
        "total_deaths",
        "outbreak_flag"
    ]
].copy()

outbreak = outbreak.rename(
    columns={
        "outbreak_count": "reported_outbreak_count",
        "total_cases": "reported_disease_cases",
        "total_deaths": "reported_disease_deaths",
        "outbreak_flag": "reported_outbreak_flag"
    }
)

# --------------------------------------------------
# MERGE DATASET 1 + FACILITY INFORMATION
# --------------------------------------------------

print("Merging Dataset 1 with facility information...")

master = df1.merge(
    facility,
    on="facility_id",
    how="left",
    suffixes=("", "_geo")
)

# --------------------------------------------------
# MERGE POPULATION
# --------------------------------------------------

print("Merging population information...")

master = master.merge(
    population,
    on="district_normalized",
    how="left"
)

# --------------------------------------------------
# MERGE DISEASE SIGNALS
# --------------------------------------------------

print("Merging disease/outbreak signals...")

master = master.merge(
    outbreak,
    left_on=["district_normalized", "month"],
    right_on=["district_normalized", "report_month"],
    how="left"
)

master = master.drop(columns=["report_month"])

# --------------------------------------------------
# FILL ABSENT OUTBREAK SIGNALS WITH ZERO
# --------------------------------------------------

master["reported_outbreak_count"] = (
    master["reported_outbreak_count"].fillna(0)
)

master["reported_disease_cases"] = (
    master["reported_disease_cases"].fillna(0)
)

master["reported_disease_deaths"] = (
    master["reported_disease_deaths"].fillna(0)
)

master["reported_outbreak_flag"] = (
    master["reported_outbreak_flag"].fillna(0)
)

# --------------------------------------------------
# COORDINATE USABILITY
# --------------------------------------------------

master["coordinate_usable"] = (
    master["match_confidence"].isin(["High", "Medium"])
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

master.to_csv(
    output_file,
    index=False
)

print()
print("===== MASTER DATASET CREATED =====")
print(f"Rows: {len(master):,}")
print(f"Columns: {len(master.columns)}")
print(f"Facilities: {master['facility_id'].nunique():,}")
print(f"Medicines: {master['medicine_id'].nunique():,}")
print(f"Districts: {master['district_normalized'].nunique()}")
print()
print("Missing district:",
      master["district_normalized"].isna().sum())

print("Missing population:",
      master["population"].isna().sum())

print("Coordinate usable:",
      master["coordinate_usable"].sum())

print("Output:")
print(output_file)