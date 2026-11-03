# R `survey` primary analysis (NHANES postpartum iron / folate)

English README for the reproducible R analysis that supports the manuscript
*Iron stores remain depleted for over a year after childbirth while red cell folate stays elevated in US women*.

This folder prepares the public GitHub + Zenodo deposit. **No GitHub URL or Zenodo DOI is claimed yet**; deposit on acceptance.

## What this reproduces

| Target | Expected (Python / manuscript) | Script |
|---|---|---|
| Body iron 1–6 mo vs comparators | −1.18 mg/kg (SE 0.439) | `01_body_iron_main.R` |
| Body iron 7–12 mo | −0.93 mg/kg (SE 0.323) | same |
| Body iron spline last significant month | month **14** | same |
| Broader cross-check (ferritin, HOMA-IR, RBC folate, sleep) | `docs/r_crosscheck.md` | `../R/01_crosscheck.R` |

Relative SE difference vs the Python pipeline was **&lt;0.01%** (max logistic SE relative difference 0.0016%); see `docs/r_crosscheck.md` and `output/r_crosscheck_comparison.csv`.

**Manuscript reporting:** primary estimates are from R package `survey`; the Python module (`code/svy.py`) runs the full pipeline and was cross-checked against R.

## Requirements

- R ≥ 4.3 with packages `survey`, `data.table`
- On this box: `Rscript` (survey 4.5)

## Input data

1. **Public NHANES XPT** (CDC / NCHS): download cycles 2007–2008 through 2017–March 2020 as in `code/01_download.sh`. Cite NCHS / NHANES; do not redistribute participant-level data beyond CDC terms.
2. Build the analytic CSV used by R:

```bash
cd .
python3 code/20_export_for_R.py   # writes data/derived/analytic_for_R.csv.gz
```

Body iron is already computed in the analytic file with the Cook equation after Roche→Flowers sTfR conversion (see Methods).

## How to run

```bash
cd .
Rscript code/r_survey/01_body_iron_main.R
# optional full cross-check suite:
Rscript code/R/01_crosscheck.R
```

Outputs land in `code/r_survey/output/`:

- `bodyiron_segments.csv` — adjusted segment contrasts
- `bodyiron_rcs_curve.csv` — month 1–24 difference vs comparators
- `bodyiron_rcs_summary.csv` — `last_sig_month` (expect 14)

## Design notes (must match Python)

- `svydesign(ids=~PSU, strata=~STR, weights=~w, nest=TRUE)`
- Multi-cycle weights: 2-year weight × 2/T; pre-pandemic (P) × 3.2/T
- Domain analysis via `subset()`; keep zero-weight rows in the design object
- Drop subsample weights ≤ 1e-6
- RCS knots: unweighted 5/35/65/95 percentiles of months among postpartum rows with weight &gt; 0
- Confidence intervals in the manuscript use **design df** (PSUs − strata); R `svyglm` residual df is slightly more conservative (documented in `docs/r_crosscheck.md`)

## Citation / deposit (placeholders — fill on acceptance)

- GitHub repository: *to be added on acceptance*
- Zenodo DOI: *to be added on acceptance*
- NHANES: National Center for Health Statistics. National Health and Nutrition Examination Survey. https://www.cdc.gov/nchs/nhanes/

## Licence for code

Analysis code in this folder is intended for release under MIT (or as required by the journal) upon acceptance. NHANES data remain under CDC / NCHS terms.
