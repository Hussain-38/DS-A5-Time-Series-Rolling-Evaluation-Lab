# DS-A5 | Time-Series Rolling Evaluation Lab

**Course:** Data Science  
**Assignment:** DS-A5  
**Research task:** One-hour-ahead forecasting of Beijing PM2.5  
**Study period:** 2010-01-01 to 2014-12-31

This is the consolidated VS Code project folder. The dataset, result tables, figures, and PDF report are kept inside this one project directory.

## Folder structure

```text
DS-A5_VSCode_Project/
├── data/                  # Original hourly PM2.5 CSV
├── scripts/               # Python scripts
├── results/
│   ├── tables/            # CSV outputs and audit tables
│   └── figures/           # PNG and PDF figures
├── reports/               # Final PDF report(s)
├── docs/                  # Method and reproducibility notes
├── .vscode/               # VS Code launch/tasks/settings
├── requirements.txt
├── environment.yml
└── README.md
```

## Open in VS Code

1. Extract this folder.
2. Open VS Code.
3. Select **File → Open Folder** and choose `DS-A5_VSCode_Project`.
4. Open the integrated terminal (**Terminal → New Terminal**).
5. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Run the two recovered scripts in order:

   ```powershell
   python scripts/01_data_inspection.py
   python scripts/02_feature_availability_map.py
   ```

The VS Code Run and Debug menu also contains launch configurations for the two recovered inspection scripts. The PDF generator is included, but requires all listed result CSVs and figures (which are included) and should be run only after confirming its paths.

## Important source-code limitation

The supplied project archive contained complete source for `01_data_inspection.py` and `02_feature_availability_map.py`; these have been changed to use paths relative to the repository root and the `data/` folder. The PDF generator was supplied separately and its project/data paths were adapted for this folder layout. Several later `.py` files in the archive were only empty Spyder header stubs (92 bytes), so their executable code was not present and has not been fabricated. The archived CSV outputs, figures, and final PDF are included. See `scripts/SOURCE_CODE_STATUS.md`.

The PDF-generation script was supplied separately and copied into `scripts/`; its local file paths should be checked against this folder before running because it originated in the Spyder/Windows project.

## Archived experimental summary

- 43,824 hourly records
- 41,757 observed PM2.5 values
- 2,067 missing PM2.5 values (4.717%)
- One-hour forecast horizon
- Six expanding-window origins
- 30-day test windows
- Models: persistence, 24-hour seasonal-naive, Random Forest

The stored results are historical outputs from the previous run. They are not evidence that the missing scripts can currently be rerun from this package.

## AI-use and verification

AI assistance was used for code troubleshooting, workflow organization, documentation, and report preparation. The researcher must independently verify all code, metrics, figures, data provenance, and scientific interpretations before submission.
