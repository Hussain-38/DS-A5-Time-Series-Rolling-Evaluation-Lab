# -*- coding: utf-8 -*-
"""
DS-A5: Generate the final academic PDF report in Spyder.

Expected project structure:
DS-A5 Time-Series Rolling Evaluation Lab/
│
├── 10_generate_final_pdf_report.py   <- save this script here
├── Dataset/
│   └── PRSA_data_2010.1.1-2014.12.31.csv
└── results/
    ├── *.csv
    └── figures/
        ├── fig_05_rolling_mean.png
        ├── fig_06_monthly_missing_data.png
        ├── fig_08_overall_model_mae.png
        ├── fig_10_mae_rolling_origins.png
        ├── fig_12_january_2013_forecast_comparison.png
        ├── fig_13_forecasting_error_over_time.png
        ├── fig_15_extreme_pollution_observations.png
        ├── fig_16_target_distribution_drift.png
        ├── fig_20_extreme_event_errors.png
        ├── fig_22_reporting_delay_mae.png
        ├── fig_23_reporting_delay_rmse.png
        └── fig_january_july_mae_boxplot.png  <- optional extra image

Install dependencies once in Anaconda Prompt:
    pip install reportlab pandas numpy pillow

Run in Spyder:
    %runfile '.../10_generate_final_pdf_report.py' --wdir

The script reads the existing results; it does not rerun forecasting experiments.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATASET_PATH = (
    PROJECT_DIR / "data" /
    "PRSA_data_2010.1.1-2014.12.31.csv"
)

OUTPUT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "DS-A5_Final_Report_With_Figure_Explanations.pdf"

print("=" * 76)
print("DS-A5: FINAL PDF REPORT GENERATION")
print("=" * 76)
print("Project:", PROJECT_DIR)
print("Results:", RESULTS_DIR)
print("Figures:", FIGURES_DIR)
print("Output:", PDF_PATH)

# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

REQUIRED_CSVS = [
    "random_forest_overall_metrics.csv",
    "random_forest_fold_metrics.csv",
    "target_distribution_drift.csv",
    "reporting_delay_overall_metrics.csv",
    "extreme_pollution_error_analysis.csv",
]

missing_files = [
    RESULTS_DIR / name for name in REQUIRED_CSVS
    if not (RESULTS_DIR / name).exists()
]

if missing_files:
    raise FileNotFoundError(
        "Required result CSV files are missing:\n" +
        "\n".join(str(p) for p in missing_files)
    )

if not FIGURES_DIR.exists():
    raise FileNotFoundError(
        f"Figure directory not found: {FIGURES_DIR}"
    )

# Dataset is used only to verify its presence and report basic coverage.
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Expected:\n" + str(DATASET_PATH)
    )

print("\n[STEP 1] Required input files verified.")

# ============================================================
# 3. LOAD EXPERIMENTAL RESULTS
# ============================================================

overall = pd.read_csv(RESULTS_DIR / "random_forest_overall_metrics.csv")
fold_metrics = pd.read_csv(RESULTS_DIR / "random_forest_fold_metrics.csv")
drift = pd.read_csv(RESULTS_DIR / "target_distribution_drift.csv")
delay = pd.read_csv(RESULTS_DIR / "reporting_delay_overall_metrics.csv")
extreme = pd.read_csv(RESULTS_DIR / "extreme_pollution_error_analysis.csv")

# Normalize common labels without altering numerical values.
overall["model"] = overall["model"].astype(str)
delay["model"] = delay["model"].astype(str)

# Basic dataset summary
dataset = pd.read_csv(DATASET_PATH)
expected_columns = {"year", "month", "day", "hour", "pm2.5"}
if not expected_columns.issubset(dataset.columns):
    raise ValueError(
        "The source dataset does not contain the expected date/target columns."
    )

dataset["datetime"] = pd.to_datetime(
    dataset[["year", "month", "day", "hour"]]
)
dataset = dataset.sort_values("datetime")

n_total = len(dataset)
n_valid = int(dataset["pm2.5"].notna().sum())
n_missing = int(dataset["pm2.5"].isna().sum())
missing_pct = 100 * n_missing / n_total

print("[STEP 2] Results and dataset loaded.")
print("Dataset observations:", n_total)
print("Valid PM2.5:", n_valid)
print("Missing PM2.5:", n_missing)

# ============================================================
# 4. CHECK REQUIRED METRICS AND EXTRACT KEY VALUES
# ============================================================

def metric_row(frame, model, scenario=None):
    subset = frame[frame["model"].str.lower() == model.lower()]
    if scenario is not None:
        subset = subset[subset["scenario"] == scenario]
    if subset.empty:
        raise ValueError(
            f"Could not find model={model}, scenario={scenario}"
        )
    return subset.iloc[0]


rf_overall = metric_row(overall, "random_forest")
persistence_overall = metric_row(overall, "persistence")
seasonal_overall = metric_row(overall, "seasonal_naive_24h")

rf_zero = metric_row(delay, "random_forest", "zero_delay")
rf_delay = metric_row(delay, "random_forest", "one_hour_delay")
pers_zero = metric_row(delay, "persistence", "zero_delay")
pers_delay = metric_row(delay, "persistence", "one_hour_delay")
seasonal_zero = metric_row(delay, "seasonal_naive_24h", "zero_delay")
seasonal_delay = metric_row(delay, "seasonal_naive_24h", "one_hour_delay")

fold3_drift = drift.loc[drift["fold"] == 3].iloc[0]
fold3_extreme_rf = extreme[
    (extreme["fold"] == 3) &
    (extreme["event_type"].str.lower() == "extreme") &
    (extreme["model"].str.lower() == "random forest")
]
fold3_extreme_persistence = extreme[
    (extreme["fold"] == 3) &
    (extreme["event_type"].str.lower() == "extreme") &
    (extreme["model"].str.lower() == "persistence")
]

# Use values from the actual CSV rather than typing them into the report.
if fold3_extreme_rf.empty or fold3_extreme_persistence.empty:
    raise ValueError("Fold 3 extreme-event metrics are missing.")

rf_extreme_mae = float(fold3_extreme_rf.iloc[0]["MAE"])
rf_extreme_bias = float(fold3_extreme_rf.iloc[0]["Bias"])
persistence_extreme_mae = float(
    fold3_extreme_persistence.iloc[0]["MAE"]
)

print("[STEP 3] Key metrics extracted from CSV files.")

# ============================================================
# 5. REPORT STYLES
# ============================================================

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="CoverTitleCustom",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=23,
    leading=28,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#16324F"),
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="CoverSubtitleCustom",
    parent=styles["Normal"],
    fontSize=10,
    leading=15,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#536779")
))

styles.add(ParagraphStyle(
    name="SectionCustom",
    parent=styles["Heading1"],
    fontSize=15,
    leading=19,
    textColor=colors.HexColor("#16324F"),
    spaceBefore=8,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="BodyCustom",
    parent=styles["BodyText"],
    fontSize=8.8,
    leading=12.5,
    spaceAfter=7
))

styles.add(ParagraphStyle(
    name="CaptionCustom",
    parent=styles["BodyText"],
    fontSize=7.7,
    leading=10,
    textColor=colors.HexColor("#424F5B"),
    spaceAfter=5
))

styles.add(ParagraphStyle(
    name="InterpretationCustom",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=12,
    leftIndent=5,
    rightIndent=5,
    borderColor=colors.HexColor("#D6DEE6"),
    borderWidth=0.5,
    borderPadding=6,
    backColor=colors.HexColor("#F6F8FA"),
    spaceAfter=9
))

styles.add(ParagraphStyle(
    name="NoteCustom",
    parent=styles["BodyText"],
    fontSize=8.3,
    leading=11.5,
    borderColor=colors.HexColor("#E6C58F"),
    borderWidth=0.5,
    borderPadding=6,
    backColor=colors.HexColor("#FFF7EA"),
    spaceAfter=8
))


def P(text, style="BodyCustom"):
    """Create a ReportLab paragraph."""
    return Paragraph(text, styles[style])


# ============================================================
# 6. TABLE HELPER
# ============================================================

def make_table(frame, columns, headings, widths):
    data = [headings]

    for _, row in frame[columns].iterrows():
        formatted = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        data.append(formatted)

    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("GRID", (0, 0), (-1, -1), 0.35,
         colors.HexColor("#C8D1D9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F6F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# ============================================================
# 7. FIGURE LIST AND SCIENTIFIC INTERPRETATIONS
# ============================================================

# All standard figure filenames below should exist in results/figures.
# If your local filename differs, edit only the relevant "file" value.
FIGURES = [
    {
        "file": "fig_08_overall_model_mae.png",
        "caption": "Figure 1. Overall MAE comparison across the three forecasting models.",
        "text": (
            f"The archived common-sample results show persistence with an "
            f"MAE of {persistence_overall['MAE']:.3f}, Random Forest with an "
            f"MAE of {rf_overall['MAE']:.3f}, and the 24-hour seasonal-naive "
            f"baseline with an MAE of {seasonal_overall['MAE']:.3f}. "
            "Persistence is the strongest aggregate reference for this "
            "one-hour-ahead task. The result is specific to this dataset, "
            "forecast horizon and rolling evaluation design; it does not "
            "establish that persistence will dominate at longer horizons."
        )
    },
    {
        "file": "fig_10_mae_rolling_origins.png",
        "caption": "Figure 2. MAE across the six expanding-window evaluation origins.",
        "text": (
            "Forecast error varies across the six test windows rather than "
            "remaining constant through time. The January 2013 fold is the "
            "most difficult period for both Random Forest and persistence. "
            "Its Random Forest MAE is approximately 27.24, compared with "
            "approximately 23.54 for persistence; July 2013 errors are "
            "substantially lower. This demonstrates why aggregate metrics "
            "alone can conceal periods of elevated forecasting risk."
        )
    },
    {
        "file": "fig_16_target_distribution_drift.png",
        "caption": "Figure 3. Target-distribution drift between training history and test windows.",
        "text": (
            f"At the January 2013 origin, the training mean is "
            f"{fold3_drift['reference_mean']:.2f} and the test mean is "
            f"{fold3_drift['comparison_mean']:.2f}; the KS statistic is "
            f"{fold3_drift['ks_statistic']:.3f}. This is the largest "
            "distributional separation across the six origins. The KS test "
            "identifies a distributional difference, but it does not reveal "
            "the physical cause of that difference."
        )
    },
    {
        "file": "fig_20_extreme_event_errors.png",
        "caption": "Figure 4. Forecasting errors during extreme pollution events.",
        "text": (
            f"In the January 2013 fold, Random Forest extreme-event MAE is "
            f"{rf_extreme_mae:.2f} and its bias is {rf_extreme_bias:.2f}, "
            f"indicating underprediction on average for the extreme subset. "
            f"Persistence extreme-event MAE is {persistence_extreme_mae:.2f}. "
            "This illustrates that good average performance does not ensure "
            "reliable forecasts during rare high-concentration episodes. "
            "The extreme subset is smaller than the full test sample, so "
            "these estimates are less stable."
        )
    },
    {
        "file": "fig_22_reporting_delay_mae.png",
        "caption": "Figure 5. MAE under zero-hour and one-hour reporting-delay scenarios.",
        "text": (
            f"On the matched sample, persistence MAE changes from "
            f"{pers_zero['MAE']:.3f} under zero delay to "
            f"{pers_delay['MAE']:.3f} under a one-hour delay. Random Forest "
            f"MAE changes from {rf_zero['MAE']:.3f} to "
            f"{rf_delay['MAE']:.3f}. This highlights the potential "
            "operational cost of delayed data. Actual sensor publication "
            "and ingestion timestamps are not present in the dataset, so "
            "these are scenario results based on stated assumptions."
        )
    },
    {
        "file": "fig_23_reporting_delay_rmse.png",
        "caption": "Figure 6. RMSE under zero-hour and one-hour reporting-delay scenarios.",
        "text": (
            f"Persistence RMSE increases from {pers_zero['RMSE']:.3f} to "
            f"{pers_delay['RMSE']:.3f}, while Random Forest RMSE increases "
            f"from {rf_zero['RMSE']:.3f} to {rf_delay['RMSE']:.3f}. "
            "The increase in RMSE indicates that larger errors contribute "
            "more strongly under the delayed-information scenario. "
            "Seasonal-naive remains unchanged in this specific experiment "
            "because of how the delayed feature scenario was constructed; "
            "this should not be generalized to all deployments."
        )
    },
    {
        "file": "fig_january_july_mae_boxplot.png",
        "caption": "Figure 7. January versus July forecasting MAE distributions.",
        "text": (
            "The boxplots show higher and more variable January errors than "
            "July errors for both persistence and Random Forest. This agrees "
            "with the fold-level evaluation, where January test windows are "
            "generally more difficult and July windows have lower errors. "
            "The plot should be interpreted according to the grouping used "
            "to construct its underlying samples; it is not automatically "
            "evidence of independent replication across years."
        )
    },
    {
        "file": "fig_05_rolling_mean.png",
        "caption": "Figure 8. Long-term PM2.5 variation represented by a 30-day rolling mean.",
        "text": (
            "The smoothed series shows strong multi-month and interannual "
            "variability rather than a simple linear trend. Elevated periods "
            "occur in several years, with a prominent peak around early 2013. "
            "These regime changes motivate rolling-origin evaluation because "
            "a model assessed under only one average regime may not represent "
            "performance during a different pollution regime. Smoothing "
            "suppresses hourly extremes, so this figure describes background "
            "variation rather than event-level exposure."
        )
    },
    {
        "file": "fig_06_monthly_missing_data.png",
        "caption": "Figure 9. Monthly percentage of missing PM2.5 observations.",
        "text": (
            f"The dataset contains {n_missing:,} missing PM2.5 values "
            f"({missing_pct:.2f}% of {n_total:,} hourly records). Missingness "
            "is temporally uneven, with some months in 2010–2012 showing "
            "large gaps and generally lower missingness in later years. "
            "The plot describes when missingness occurs, but does not establish "
            "whether missingness is random or related to pollution conditions."
        )
    },
    {
        "file": "fig_12_january_2013_forecast_comparison.png",
        "caption": "Figure 10. Actual and predicted PM2.5 during the first seven days of January 2013.",
        "text": (
            "The prediction traces broadly follow the observed temporal "
            "pattern, including the major rise around 5 January and the "
            "large peak around 7 January. Rapid transitions reveal forecast "
            "limitations: predictions can overshoot or lag abrupt changes, "
            "and peak magnitude is not perfectly reproduced. This selected "
            "seven-day example illustrates behavior during a difficult "
            "period; it is not a representative sample of every test hour."
        )
    },
    {
        "file": "fig_13_forecasting_error_over_time.png",
        "caption": "Figure 11. Forecasting error over time by evaluation fold.",
        "text": (
            "Error spikes are concentrated in short episodes rather than "
            "being uniformly distributed through each test window. The "
            "largest visible Random Forest spike occurs in the January 2013 "
            "window, consistent with the high fold-level error. Persistence "
            "also experiences spikes, showing that the critical period is "
            "difficult for both a simple baseline and the learned model. "
            "A rolling error curve can support monitoring, but operational "
            "thresholds require separate validation."
        )
    },
    {
        "file": "fig_15_extreme_pollution_observations.png",
        "caption": "Figure 12. Extreme PM2.5 observations through time and the 99th-percentile threshold.",
        "text": (
            "The plot identifies upper-tail PM2.5 observations using the "
            "99th-percentile threshold shown in the figure. Extreme values "
            "occur in temporal clusters rather than being evenly distributed. "
            "Such clustering supports separate evaluation of high-concentration "
            "events rather than relying only on average MAE. The percentile "
            "threshold is descriptive of this sample and is not a health-based "
            "regulatory threshold."
        )
    },
]


# ============================================================
# 8. BUILD THE PDF CONTENT
# ============================================================

story = []

# Cover
story.extend([
    Spacer(1, 30 * mm),
    P("TIME-SERIES ROLLING<br/>EVALUATION LAB", "CoverTitleCustom"),
    P(
        "DS-A5 | Final report with figure-by-figure scientific interpretation",
        "CoverSubtitleCustom"
    ),
    Spacer(1, 8 * mm),
    P("<b>One-hour-ahead forecasting of Beijing PM2.5</b>",
      "CoverSubtitleCustom"),
    Spacer(1, 15 * mm),
    P(
        "Dataset: hourly observations from 1 January 2010 to "
        "31 December 2014<br/>"
        "Evaluation: six expanding-window origins; 30-day test windows<br/>"
        "Models: persistence, 24-hour seasonal-naive and Random Forest",
        "CoverSubtitleCustom"
    ),
    Spacer(1, 18 * mm),
    P(
        "This report synthesizes the supplied dataset, archived evaluation "
        "tables and generated figures. Interpretations are based on the "
        "available experimental outputs. This script does not rerun the "
        "forecasting experiments.",
        "CoverSubtitleCustom"
    ),
    PageBreak()
])

# Executive summary
story.extend([
    P("Executive summary", "SectionCustom"),
    P(
        f"The archived evaluation compares persistence, a 24-hour "
        f"seasonal-naive baseline and Random Forest for one-hour-ahead "
        f"PM2.5 forecasting. On the common sample, persistence achieved "
        f"the lowest overall MAE ({persistence_overall['MAE']:.3f}) and "
        f"RMSE ({persistence_overall['RMSE']:.3f}). Random Forest achieved "
        f"MAE {rf_overall['MAE']:.3f} and RMSE {rf_overall['RMSE']:.3f}. "
        f"The 24-hour seasonal-naive baseline performed substantially "
        f"worse (MAE {seasonal_overall['MAE']:.3f}; "
        f"RMSE {seasonal_overall['RMSE']:.3f}).",
        "BodyCustom"
    ),
    P(
        f"Performance varied across the six origins. January 2013 was the "
        f"clearest critical period: the test-period mean was "
        f"{fold3_drift['comparison_mean']:.2f}, compared with a historical "
        f"training mean of {fold3_drift['reference_mean']:.2f}; the KS "
        f"statistic was {fold3_drift['ks_statistic']:.3f}. This illustrates "
        f"why a single aggregate score is insufficient for evaluating "
        f"forecasting under temporal drift.",
        "BodyCustom"
    ),
    P(
        "The one-hour reporting-delay experiment is a scenario analysis. "
        "The source dataset does not contain actual sensor release or "
        "ingestion timestamps; therefore, the delay assumptions are not "
        "verified operational timestamps.",
        "NoteCustom"
    )
])

# Dataset and evaluation
story.extend([
    P("1. Dataset and temporal evaluation", "SectionCustom"),
    P(
        f"The dataset contains {n_total:,} hourly records, of which "
        f"{n_valid:,} have an observed PM2.5 target and {n_missing:,} "
        f"are missing ({missing_pct:.3f}%). The series spans "
        f"{dataset['datetime'].min()} to {dataset['datetime'].max()}. "
        "The rolling evaluation uses six expanding training windows, "
        "30-day test periods and a one-hour forecast horizon. Missing "
        "target values are not imputed in the reported evaluation; "
        "comparisons use valid paired observations.",
        "BodyCustom"
    ),
    P("2. Overall model performance", "SectionCustom"),
    make_table(
        overall,
        ["model", "N", "MAE", "RMSE", "Bias"],
        ["Model", "N", "MAE", "RMSE", "Bias"],
        [43 * mm, 17 * mm, 23 * mm, 23 * mm, 23 * mm]
    ),
    Spacer(1, 3 * mm),
    P(
        "Persistence is the strongest aggregate reference. Random Forest "
        "improves substantially over the seasonal-naive baseline but does "
        "not outperform persistence overall. This is a meaningful negative "
        "result: the nonlinear model and additional predictors did not "
        "produce an overall gain over the latest observed PM2.5 value for "
        "this one-hour horizon.",
        "BodyCustom"
    ),
    P("3. Drift and critical-period evidence", "SectionCustom"),
    make_table(
        drift,
        ["fold", "forecast_origin", "reference_mean",
         "comparison_mean", "ks_statistic", "ks_p_value"],
        ["Fold", "Origin", "Training mean", "Test mean", "KS D", "p-value"],
        [13 * mm, 27 * mm, 27 * mm, 27 * mm, 18 * mm, 24 * mm]
    ),
    Spacer(1, 3 * mm),
    P(
        "The January 2013 test window stands out in both mean level and "
        "distributional separation. The KS test detects a distributional "
        "difference, but it does not identify the physical cause of the "
        "change and should not be interpreted as proof of a particular "
        "emissions or meteorological mechanism.",
        "BodyCustom"
    ),
    P("4. Reporting-delay sensitivity", "SectionCustom"),
    make_table(
        delay,
        ["scenario", "reporting_delay_hours", "model", "N", "MAE", "RMSE"],
        ["Scenario", "Delay (h)", "Model", "N", "MAE", "RMSE"],
        [29 * mm, 18 * mm, 39 * mm, 14 * mm, 23 * mm, 23 * mm]
    ),
    Spacer(1, 3 * mm),
    P(
        "On the matched sample, one-hour delay increased persistence and "
        "Random Forest errors. Seasonal-naive remains unchanged in this "
        "specific experiment because the constructed delay scenario does "
        "not alter its 24-hour reference in the same way. This is a result "
        "of the experiment design, not evidence that seasonal-naive is "
        "generally immune to data latency.",
        "BodyCustom"
    ),
    PageBreak(),
    P("5. Figure-by-figure results and interpretation", "SectionCustom"),
    P(
        "Each figure is followed by a scientific interpretation describing "
        "the visible pattern, its relationship to the quantitative results, "
        "and the main limits on inference.",
        "BodyCustom"
    )
])


def add_figure(figure_info, figure_number, total_figures):
    image_path = FIGURES_DIR / figure_info["file"]

    if not image_path.exists():
        print("[WARNING] Figure not found and will be skipped:", image_path)
        return False

    image = RLImage(str(image_path))
    max_width = 170 * mm
    max_height = 105 * mm

    scale = min(
        max_width / image.imageWidth,
        max_height / image.imageHeight
    )

    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale

    block = [
        image,
        Spacer(1, 2 * mm),
        P(figure_info["caption"], "CaptionCustom"),
        P(
            "<b>Interpretation.</b> " + figure_info["text"],
            "InterpretationCustom"
        )
    ]

    story.extend(block)

    if figure_number < total_figures:
        story.append(PageBreak())

    return True


print("\n[STEP 4] Adding figures and explanations.")

included_figures = 0
for i, fig in enumerate(FIGURES, start=1):
    if add_figure(fig, i, len(FIGURES)):
        included_figures += 1

if included_figures == 0:
    raise RuntimeError("No figures were added. Check the figure directory.")

# Monitoring and limitations
story.extend([
    PageBreak(),
    P("6. Monitoring, retraining and fallback plan", "SectionCustom"),
    P(
        "Monitor forecast MAE and RMSE over rolling windows, signed bias, "
        "missing-target rate, input-feature availability, target-distribution "
        "summaries and the frequency and magnitude of extreme events. "
        "Compare the deployed model with persistence on the same timestamps. "
        "A sustained increase in error, a material distribution shift, or "
        "repeated underprediction during high-PM2.5 episodes should trigger "
        "investigation rather than automatic retraining alone.",
        "BodyCustom"
    ),
    P(
        "Retraining should be considered when deterioration persists across "
        "more than one monitoring window and when incoming data are verified "
        "as available at the forecast origin. Refit preprocessing and model "
        "components using only data available up to the retraining cutoff. "
        "Until a model is revalidated, persistence is a transparent fallback "
        "because it was the strongest aggregate model in this experiment. "
        "For high-pollution warnings, add a separate event-focused safeguard "
        "and do not rely solely on average MAE.",
        "BodyCustom"
    ),
    P("7. Limitations and reproducibility", "SectionCustom"),
    P(
        "Actual sensor release and ingestion timestamps are not included in "
        "the source dataset. Consequently, the zero-hour and one-hour "
        "availability settings are explicit scenarios, not verified "
        "operational timestamps. This PDF script reads archived CSV results "
        "and figures; it does not rerun model training or independently "
        "validate the full forecasting pipeline. Ensure that the complete "
        "Python source scripts, package versions and dataset provenance are "
        "preserved with the repository before submission.",
        "NoteCustom"
    ),
    P("8. AI-use disclosure", "SectionCustom"),
    P(
        "ChatGPT was used to assist with workflow planning, debugging based "
        "on user-provided console output, organization of result tables and "
        "drafting figure interpretations. Numerical values in this report "
        "are read from the archived CSV outputs. The author should "
        "independently verify code, figure provenance, calculations and "
        "the final repository contents before submission.",
        "BodyCustom"
    )
])


# ============================================================
# 9. PAGE HEADER AND FOOTER
# ============================================================

def add_page_footer(canvas, doc):
    canvas.saveState()

    page_width, page_height = A4

    canvas.setStrokeColor(colors.HexColor("#D5DDE5"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#657585"))
    canvas.drawString(
        18 * mm, 10 * mm,
        "DS-A5 | Time-Series Rolling Evaluation Lab"
    )
    canvas.drawRightString(
        page_width - 18 * mm, 10 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# 10. GENERATE PDF
# ============================================================

print("\n[STEP 5] Building PDF...")

document = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=17 * mm,
    bottomMargin=22 * mm,
    title="DS-A5 Final Report: Time-Series Rolling Evaluation",
    author="DS-A5 Project",
    subject="One-hour-ahead Beijing PM2.5 forecasting"
)

document.build(
    story,
    onFirstPage=add_page_footer,
    onLaterPages=add_page_footer
)

if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
    raise RuntimeError("PDF generation failed.")

print("\n" + "=" * 76)
print("PDF REPORT GENERATED SUCCESSFULLY")
print("=" * 76)
print("Output:", PDF_PATH)
print("Figures included:", included_figures, "/", len(FIGURES))
print("File size: {:.2f} MB".format(PDF_PATH.stat().st_size / (1024 ** 2)))
print("\nIMPORTANT:")
print("- The optional January-versus-July boxplot must be saved as:")
print("  results/figures/fig_january_july_mae_boxplot.png")
print("- Missing figures are reported as warnings and skipped.")
print("- This script creates the PDF from existing CSVs and figures.")
print("- It does not rerun the forecasting experiments.")
print("=" * 76)
