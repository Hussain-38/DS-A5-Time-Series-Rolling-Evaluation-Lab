
# ================================================================
# DS-A5: TIME-SERIES ROLLING EVALUATION LAB
# Script 01: Initial Dataset Inspection
#
# Dataset: Beijing PM2.5 (2010-2014)
# Environment: Python / Spyder
#
# IMPORTANT:
# Original CSV is located directly inside the project folder.
# No modifications are made to the original dataset.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np

from pathlib import Path


print("\n" + "=" * 70)
print("DS-A5: BEIJING PM2.5 DATASET INSPECTION")
print("=" * 70)



# ================================================================
# 2. DEFINE PROJECT AND DATA PATHS
# ================================================================

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"

print("\n[STEP 1] PROJECT DIRECTORY")

print("Project directory:", PROJECT_DIR)

# Verify project directory

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist: {PROJECT_DIR}"
    )

# Search for the CSV anywhere inside the project directory

csv_files = list(
    DATA_DIR.glob("PRSA_data_2010.1.1-2014.12.31.csv")
)

print("\n[STEP 2] SEARCHING FOR DATASET")

if len(csv_files) == 0:

    print("\nNo matching CSV file found.")

    print("\nFiles currently present in project directory:")

    for file in PROJECT_DIR.iterdir():
        print(file.name)

    raise FileNotFoundError(
        "\nThe CSV file is not present inside the project directory."
        "\nPlease locate the original CSV file in Windows Explorer."
    )

elif len(csv_files) > 1:

    print("\nMultiple matching CSV files found:")

    for file in csv_files:
        print(file)

    raise RuntimeError(
        "Multiple copies found. Please keep one clearly identified dataset."
    )

else:

    DATA_PATH = csv_files[0]

    print("\nDataset successfully located!")

    print("Dataset path:", DATA_PATH)

    print("File size (MB):", round(DATA_PATH.stat().st_size / (1024**2), 3))


# ================================================================
# 3. LOAD ORIGINAL DATASET
# ================================================================

print("\n[STEP 2] LOADING DATASET")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

print("Number of observations:", df.shape[0])

print("Number of original variables:", df.shape[1])


# ================================================================
# 4. INSPECT COLUMN NAMES
# ================================================================

print("\n[STEP 3] COLUMN INFORMATION")

for i, column in enumerate(df.columns, start=1):

    print(f"{i:02d}. {column}")


# ================================================================
# 5. INSPECT DATA TYPES
# ================================================================

print("\n[STEP 4] DATA TYPES")

print(df.dtypes)


# ================================================================
# 6. DISPLAY FIRST AND LAST OBSERVATIONS
# ================================================================

print("\n[STEP 5] FIRST FIVE OBSERVATIONS")

print(df.head())


print("\nLAST FIVE OBSERVATIONS")

print(df.tail())


# ================================================================
# 7. CHECK MISSING VALUES
# ================================================================

print("\n[STEP 6] MISSING VALUE ANALYSIS")

missing_summary = pd.DataFrame({

    "Missing_Count": df.isnull().sum(),

    "Missing_Percentage": df.isnull().mean() * 100

})

print(missing_summary.round(3).to_string())

print("\nTotal missing values:", df.isnull().sum().sum())


# ================================================================
# 8. CONSTRUCT DATETIME INDEX
# ================================================================

print("\n[STEP 7] DATETIME CONSTRUCTION")

required_time_columns = [
    "year",
    "month",
    "day",
    "hour"
]

for col in required_time_columns:

    if col not in df.columns:

        raise ValueError(
            f"Required time column missing: {col}"
        )


df["datetime"] = pd.to_datetime(

    df[required_time_columns],

    errors="coerce"

)

print("\nFirst five timestamps:")

print(df["datetime"].head())

print("\nLast five timestamps:")

print(df["datetime"].tail())


invalid_dates = df["datetime"].isna().sum()

print("\nInvalid timestamps:", invalid_dates)


# ================================================================
# 9. CHECK CHRONOLOGICAL ORDER AND DUPLICATES
# ================================================================

print("\n[STEP 8] CHRONOLOGICAL VALIDATION")

is_chronological = df["datetime"].is_monotonic_increasing

duplicate_timestamps = df["datetime"].duplicated().sum()

print("Chronological order:", is_chronological)

print("Duplicate timestamps:", duplicate_timestamps)


# ================================================================
# 10. CHECK HOURLY FREQUENCY
# ================================================================

print("\n[STEP 9] HOURLY FREQUENCY CHECK")

if invalid_dates == 0:

    time_difference = df["datetime"].diff()

    incorrect_intervals = (

        time_difference.notna()

        & (time_difference != pd.Timedelta(hours=1))

    )

    print(
        "Irregular hourly intervals:",
        incorrect_intervals.sum()
    )

else:

    print("Skipped because invalid timestamps exist.")


# ================================================================
# 11. ANNUAL PM2.5 DATA AVAILABILITY
# ================================================================

print("\n[STEP 10] ANNUAL DATA AVAILABILITY")

if "pm2.5" not in df.columns:

    raise ValueError("Target variable pm2.5 not found.")


annual_summary = df.groupby("year").agg(

    Total_Observations=("pm2.5", "size"),

    Available_PM25=("pm2.5", "count"),

    Missing_PM25=("pm2.5", lambda x: x.isna().sum()),

    Mean_PM25=("pm2.5", "mean"),

    Std_PM25=("pm2.5", "std"),

    Minimum_PM25=("pm2.5", "min"),

    Maximum_PM25=("pm2.5", "max")

)

annual_summary["Missing_Percentage"] = (

    annual_summary["Missing_PM25"]

    / annual_summary["Total_Observations"]

    * 100

)

print(annual_summary.round(3).to_string())


# ================================================================
# 12. NUMERICAL DESCRIPTIVE STATISTICS
# ================================================================

print("\n[STEP 11] DESCRIPTIVE STATISTICS")

numerical_columns = df.select_dtypes(

    include=[np.number]

).columns

descriptive_statistics = df[

    numerical_columns

].describe().T

print(descriptive_statistics.round(3).to_string())


# ================================================================
# 13. CATEGORICAL VARIABLE ANALYSIS
# ================================================================

print("\n[STEP 12] CATEGORICAL VARIABLES")

categorical_columns = df.select_dtypes(

    include=["object", "category"]

).columns

for column in categorical_columns:

    print(f"\nVariable: {column}")

    print(df[column].value_counts(dropna=False))


# ================================================================
# 14. TARGET VARIABLE INSPECTION
# ================================================================

print("\n[STEP 13] PM2.5 TARGET VARIABLE")

pm25 = df["pm2.5"]

print("Available observations:", pm25.count())

print("Missing observations:", pm25.isna().sum())

print("\nDescriptive statistics:")

print(pm25.describe().round(3))


# ================================================================
# 15. BASIC DATA INTEGRITY ASSERTIONS
# ================================================================

print("\n[STEP 14] DATA INTEGRITY ASSERTIONS")

assert len(df) > 0

assert df.columns.is_unique

assert invalid_dates == 0

assert duplicate_timestamps == 0

assert is_chronological

assert df["datetime"].notna().all()

assert df["pm2.5"].isna().sum() > 0

print("All basic data integrity assertions passed.")


# ================================================================
# 16. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)

print("FINAL DATASET INSPECTION SUMMARY")

print("=" * 70)

print("Dataset shape:", df.shape)

print("Start date:", df["datetime"].min())

print("End date:", df["datetime"].max())

print("Total observations:", len(df))

print("Original variables:", len(df.columns) - 1)

print("Missing PM2.5:", df["pm2.5"].isna().sum())

print("Duplicate timestamps:", duplicate_timestamps)

print("Chronological order:", is_chronological)

print("Target variable: pm2.5")

print("Temporal frequency: Hourly")

print("\nDATASET INSPECTION COMPLETED SUCCESSFULLY")

print("=" * 70)
