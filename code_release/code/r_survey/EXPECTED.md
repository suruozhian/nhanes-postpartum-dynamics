# Expected numeric anchors (do not invent new manuscript numbers)

From a successful run of `01_body_iron_main.R` on `data/derived/analytic_for_R.csv.gz` (2026-10-29):

| Quantity | Value |
|---|---|
| Body iron seg 0–6 b (SE) | −1.17793 (0.4387) |
| Body iron seg 7–12 b (SE) | −0.92809 (0.3229) |
| Body iron seg 13–24 b (SE) | −0.34671 (0.2387) |
| Wald 3-df p (design df) | 0.01248 |
| RCS knots (months) | 1 / 8 / 14 / 23 |
| last_sig_month | **14** |
| returns_to_ctrl_from_month | 15 |

These match `docs/r_crosscheck.md` and `output/step1/T3b_rcs_tests.csv` (Body iron).
