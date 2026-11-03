# Postpartum iron stores and red cell folate in US women (NHANES) — analysis code

Code for: "Lower iron stores and higher red cell folate in US women during the first year after
childbirth: a cross-sectional NHANES comparison" (submitted to *European Journal of Nutrition*).

Licence: MIT (see `LICENSE`). Data are **not** included; all inputs are public NHANES files
downloaded from the CDC/NCHS website by `01_download_nhanes.sh`.

## Requirements
- Python 3.13.5 with the packages in `requirements.txt` (`pip install -r requirements.txt`)
- R 4.5.3 with `survey` 4.5 and `data.table` (used for the independent R cross-check)
- `bash`, `curl`, `pdftops` (poppler-utils; EPS export of figures)

## Run
```bash
bash run_all.sh          # download (~175 MB) + full pipeline
```
All paths are relative to this folder.

## Run order (what `run_all.sh` does)
| Step | Script | Purpose |
|---|---|---|
| 1 | `01_download_nhanes.sh` | Download NHANES XPT files to `data/raw/` (log: `data/download_log.tsv`) |
| 2 | `code/02_step0_counts.py` | Eligibility counts, months since delivery by cycle |
| 3 | `code/10_step1_build.py` | Analytic dataset (`data/step1_analytic.pkl`) and survey weights |
| 4 | `code/11_step1_models.py` | Segment contrasts, splines, binary outcomes, FDR |
| 5 | `code/12_step1_modifiers_sensitivity.py` | Effect modification, sensitivity analyses S1–S6 |
| 6 | `code/14_S1_residual_df.py`, `code/13_step1_draft_diagnostics.py` | S1 residual-df p values (Supplementary Table S11); diagnostics |
| 7 | `code/20_export_for_R.py`, `code/21_py_reference_for_R.py` | Export for R cross-check |
| 8 | `code/30_deep_dive.py` | Iron store package, folate mechanism, waist/breastfeeding |
| 9 | `code/40_figures.py` | Draft figures |
| 10 | `code/60_iron_health_links.py` | Appendix analyses |
| 11 | `code/70_reviewer_audit_recompute.py`, `code/71_revision_R1R8.py` | Simultaneous bands, knot sensitivity, reliability flags, S7, season |
| 12 | `code/73_figures_v2.py`, `code/76_figures_v3.py` | Final figures (`output/figures_v3/`, 600 dpi TIFF/PNG + PDF/EPS) |
| 13 | `code/r_survey/01_body_iron_main.R`, `code/R/01_crosscheck.R` | Independent R `survey` replication of main estimates |

`code/svy.py` implements Taylor-linearised design-based variance (strata/PSU) used by the Python models.

## Notes
- Design degrees of freedom follow NHANES strata/PSU structure (e.g. 72 for iron cycles).
- Random seeds are fixed in scripts that use simulation (sup-t bands).
