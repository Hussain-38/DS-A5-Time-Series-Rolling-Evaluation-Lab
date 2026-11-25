# Reproducibility checklist

- [ ] Confirm Python and package versions in the environment used for the original experiments.
- [ ] Recover missing executable scripts listed in `scripts/SOURCE_CODE_STATUS.md`.
- [ ] Inspect all absolute paths in `10_generate_final_pdf_report.py` and replace with paths based on `Path(__file__)`.
- [ ] Run the full workflow from a clean VS Code environment.
- [ ] Verify raw predictions, fold metrics, drift tables, and reporting-delay results.
- [ ] Regenerate all figures from raw results.
- [ ] Compare regenerated outputs with the archived outputs.
- [ ] Verify PDF text, figures, and claims against CSV results.
- [ ] Add a Git commit hash or release tag after validation.
