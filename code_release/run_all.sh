#!/usr/bin/env bash
# Full pipeline from raw NHANES files to all estimates and figures. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw data/derived output/{audit,deep,diagnostic,figures,figures_v2,figures_v3,iron_links,revision,step1} code/r_survey/output
bash 01_download_nhanes.sh
for s in 02_step0_counts 10_step1_build 11_step1_models \
         12_step1_modifiers_sensitivity 14_S1_residual_df 13_step1_draft_diagnostics 20_export_for_R 21_py_reference_for_R \
         30_deep_dive 40_figures 60_iron_health_links 70_reviewer_audit_recompute 71_revision_R1R8 \
         73_figures_v2 76_figures_v3; do
  echo "== $s"; PYTHONPATH=code python3 "code/$s.py"
done
Rscript code/r_survey/01_body_iron_main.R
Rscript code/R/01_crosscheck.R
echo "Done. Results: output/ ; figures: output/figures_v3/"
