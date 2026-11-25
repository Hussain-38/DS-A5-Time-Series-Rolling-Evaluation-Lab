# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:24:12 2026

@author: hussa
"""


# ================================================================
# DS-A5: TIME-SERIES ROLLING EVALUATION LAB
# Script 02: Feature Availability and Temporal Information Map
#
# Dataset: Beijing PM2.5 (2010-2014)
# Environment: Python / Spyder
#
# Objectives:
# 1. Document every original variable.
# 2. Identify its forecasting role.
# 3. Define observation and assumed availability times.
# 4. Identify potential temporal leakage.
# 5. Document label availability.
# 6. Create a one-hour reporting-delay sensitivity scenario.
# 7. Export the feature-availability map.
#
# IMPORTANT:
# The dataset does not contain actual sensor release or
# ingestion timestamps. Availability times below are explicit
# methodological assumptions, not verified operational times.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np

from pathlib import Path


print("\n" + "=" * 75)
print("DS-A5: FEATURE AVAILABILITY AND TEMPORAL INFORMATION MAP")
print("=" * 75)


# ================================================================
# 2. DEFINE PROJECT PATHS
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "PRSA_data_2010.1.1-2014.12.31.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n[STEP 1] LOADING DATASET")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# ================================================================
# 4. CONSTRUCT DATETIME
# ================================================================

print("\n[STEP 2] CONSTRUCTING DATETIME")

time_columns = [
    "year",
    "month",
    "day",
    "hour"
]

df["datetime"] = pd.to_datetime(
    df[time_columns],
    errors="coerce"
)

assert df["datetime"].notna().all()

assert df["datetime"].is_monotonic_increasing

assert df["datetime"].is_unique

print("Datetime successfully constructed.")

print("Start:", df["datetime"].min())

print("End:", df["datetime"].max())


# ================================================================
# 5. DEFINE FEATURE AND LABEL AVAILABILITY INFORMATION
# ================================================================

print("\n[STEP 3] DEFINING FEATURE AVAILABILITY")

# The availability conventions are methodological assumptions.
#
# Primary scenario:
# Measurements become available at the end of their
# observation hour.
#
# Sensitivity scenario:
# Measurements become available one hour after their
# observation hour.
#
# Actual sensor reporting times are not provided in this dataset.


availability_information = [

    {
        "variable": "No",
        "description": "Sequential row identifier",
        "role": "Identifier",
        "observation_time": "Not applicable",
        "primary_availability": "Not applicable",
        "availability_delay_hours": np.nan,
        "prediction_use": "Exclude",
        "leakage_risk": "May encode chronological position",
        "availability_evidence": "Dataset field; not an environmental measurement"
    },

    {
        "variable": "year",
        "description": "Observation year",
        "role": "Calendar information",
        "observation_time": "Timestamp metadata",
        "primary_availability": "Known in advance from calendar",
        "availability_delay_hours": 0,
        "prediction_use": "Potential calendar feature",
        "leakage_risk": "Low if derived from forecast target time",
        "availability_evidence": "Deterministic calendar information"
    },

    {
        "variable": "month",
        "description": "Observation month",
        "role": "Calendar information",
        "observation_time": "Timestamp metadata",
        "primary_availability": "Known in advance from calendar",
        "availability_delay_hours": 0,
        "prediction_use": "Potential calendar feature",
        "leakage_risk": "Low if derived from forecast target time",
        "availability_evidence": "Deterministic calendar information"
    },

    {
        "variable": "day",
        "description": "Observation day",
        "role": "Calendar information",
        "observation_time": "Timestamp metadata",
        "primary_availability": "Known in advance from calendar",
        "availability_delay_hours": 0,
        "prediction_use": "Potential calendar feature",
        "leakage_risk": "Low if derived from forecast target time",
        "availability_evidence": "Deterministic calendar information"
    },

    {
        "variable": "hour",
        "description": "Observation hour",
        "role": "Calendar information",
        "observation_time": "Timestamp metadata",
        "primary_availability": "Known in advance from calendar",
        "availability_delay_hours": 0,
        "prediction_use": "Potential calendar feature",
        "leakage_risk": "Low if derived from forecast target time",
        "availability_evidence": "Deterministic calendar information"
    },

    {
        "variable": "pm2.5",
        "description": "PM2.5 concentration (ug/m3)",
        "role": "Forecast target and historical predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical lags only; future value is label",
        "leakage_risk": "Very high if future PM2.5 enters predictors",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "DEWP",
        "description": "Dew point temperature",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "TEMP",
        "description": "Air temperature",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "PRES",
        "description": "Atmospheric pressure",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "cbwd",
        "description": "Combined wind direction",
        "role": "Categorical meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "Iws",
        "description": "Cumulative wind speed",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "Is",
        "description": "Cumulative hours of snow",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    },

    {
        "variable": "Ir",
        "description": "Cumulative hours of rain",
        "role": "Meteorological predictor",
        "observation_time": "Measurement timestamp",
        "primary_availability": "End of observation hour",
        "availability_delay_hours": 0,
        "prediction_use": "Historical observations",
        "leakage_risk": "High if future measurements are used",
        "availability_evidence": "Observation timestamp only; release time undocumented"
    }

]


availability_map = pd.DataFrame(availability_information)


# ================================================================
# 6. ADD DATASET OBSERVATION AND AVAILABILITY INFORMATION
# ================================================================

print("\n[STEP 4] DOCUMENTING AVAILABILITY RULES")

availability_map["dataset_start"] = str(
    df["datetime"].min()
)

availability_map["dataset_end"] = str(
    df["datetime"].max()
)

availability_map["frequency"] = "Hourly"

availability_map["forecast_horizon"] = "1 hour ahead"

availability_map["primary_assumption"] = (
    "Available at end of observation hour; "
    "not independently verified"
)

availability_map["sensitivity_assumption"] = (
    "Available one hour after observation timestamp"
)


# ================================================================
# 7. IDENTIFY TARGET LABEL AVAILABILITY
# ================================================================

print("\n[STEP 5] LABEL AVAILABILITY")

label_information = pd.DataFrame({

    "information": [
        "Forecast target",
        "Target observation time",
        "Primary label availability",
        "Sensitivity label availability",
        "Forecasting horizon",
        "Training label restriction"
    ],

    "definition": [
        "PM2.5 concentration",
        "t + 1 hour",
        "End of target observation hour",
        "One hour after target observation hour",
        "1 hour",
        "Label observation time must not exceed forecast origin"
    ]

})

print(label_information.to_string(index=False))


# ================================================================
# 8. CREATE A ONE-HOUR REPORTING-DELAY SCENARIO
# ================================================================

print("\n[STEP 6] REPORTING-DELAY SENSITIVITY")

measurement_variables = [

    "pm2.5",
    "DEWP",
    "TEMP",
    "PRES",
    "cbwd",
    "Iws",
    "Is",
    "Ir"

]

delay_summary = pd.DataFrame({

    "variable": measurement_variables,

    "observation_timestamp": [
        "t" for _ in measurement_variables
    ],

    "primary_availability": [
        "End of hour t" for _ in measurement_variables
    ],

    "delayed_availability": [
        "End of hour t+1" for _ in measurement_variables
    ],

    "primary_assumption": [
        "0-hour reporting delay" for _ in measurement_variables
    ],

    "sensitivity_assumption": [
        "1-hour reporting delay" for _ in measurement_variables
    ]

})

print(delay_summary.to_string(index=False))


# ================================================================
# 9. CREATE A FORECAST ORIGIN EXAMPLE
# ================================================================

print("\n[STEP 7] FORECAST ORIGIN EXAMPLE")

forecast_origin = pd.Timestamp("2014-12-31 10:00:00")

target_time = forecast_origin + pd.Timedelta(hours=1)

example = pd.DataFrame({

    "item": [
        "Historical observation at 09:00",
        "Observation at forecast origin 10:00",
        "Forecast target at 11:00"
    ],

    "timestamp": [
        forecast_origin - pd.Timedelta(hours=1),
        forecast_origin,
        target_time
    ],

    "role": [
        "Historical input",
        "Latest permitted input under primary assumption",
        "Future prediction target"
    ],

    "permitted_at_origin": [
        "Yes",
        "Yes, subject to stated availability assumption",
        "No"
    ]

})

print(example.to_string(index=False))


# ================================================================
# 10. SAVE FEATURE AVAILABILITY MAP
# ================================================================

print("\n[STEP 8] SAVING OUTPUTS")

map_path = (
    RESULTS_DIR /
    "feature_availability_map.csv"
)

availability_map.to_csv(
    map_path,
    index=False
)


label_path = (
    RESULTS_DIR /
    "label_availability_map.csv"
)

label_information.to_csv(
    label_path,
    index=False
)


delay_path = (
    RESULTS_DIR /
    "availability_delay_sensitivity.csv"
)

delay_summary.to_csv(
    delay_path,
    index=False
)


example_path = (
    RESULTS_DIR /
    "forecast_origin_example.csv"
)

example.to_csv(
    example_path,
    index=False
)


# ================================================================
# 11. TEMPORAL INFORMATION BOUNDARY ASSERTIONS
# ================================================================

print("\n[STEP 9] TEMPORAL BOUNDARY ASSERTIONS")

# Check forecast target occurs after forecast origin

assert target_time > forecast_origin

# Check all historical inputs occur no later than forecast origin

historical_input_times = [

    forecast_origin - pd.Timedelta(hours=1),

    forecast_origin

]

assert all(
    timestamp <= forecast_origin
    for timestamp in historical_input_times
)

# Check target is not included among permitted historical inputs

assert target_time not in historical_input_times

# Check all original variables are documented

assert set(df.columns) - {"datetime"} == set(
    availability_map["variable"]
)

print("All temporal boundary assertions passed.")


# ================================================================
# 12. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)

print("FEATURE AVAILABILITY MAP COMPLETED")

print("=" * 75)

print("Documented original variables:", len(availability_map))

print("Forecast target: pm2.5")

print("Forecast horizon: 1 hour")

print("Primary availability assumption: End of observation hour")

print("Sensitivity scenario: 1-hour reporting delay")

print("\nGenerated files:")

print(map_path)

print(label_path)

print(delay_path)

print(example_path)

print("\nIMPORTANT METHODOLOGICAL LIMITATION:")

print(
    "Actual sensor release and ingestion timestamps "
    "are not available in the original dataset."
)

print(
    "Availability times are documented assumptions, "
    "not verified operational timestamps."
)

print("\nSTAGE 2 SCRIPT EXECUTED SUCCESSFULLY")

print("=" * 75)
