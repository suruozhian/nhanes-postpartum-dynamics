## Primary R survey estimates for body iron (Cook equation):
##   (1) three postpartum segments vs comparators
##   (2) restricted cubic spline: difference vs comparators by month,
##       last month with a significant difference (CI excludes 0)
##
## Matches docs/r_crosscheck.md and output/step1/T3b_rcs_tests.csv
## (Body iron last_sig_month = 14).
##
## Usage (from repo root):
##   Rscript code/r_survey/01_body_iron_main.R
## Input: data/derived/analytic_for_R.csv.gz  (build with code/20_export_for_R.py)

suppressMessages({ library(survey); library(data.table) })
options(survey.lonely.psu = "fail")

ROOT <- Sys.getenv("NHANES_PP_ROOT", unset = ".")
IN   <- file.path(ROOT, "data/derived/analytic_for_R.csv.gz")
OUTD <- file.path(ROOT, "code/r_survey/output")
dir.create(OUTD, showWarnings = FALSE, recursive = TRUE)

D <- as.data.table(read.csv(gzfile(IN), stringsAsFactors = FALSE))
YRS <- c(E = 2, F = 2, G = 2, H = 2, I = 2, P = 3.2, J = 2)

mkw <- function(d, wt, cyc) {
  Tt <- sum(YRS[cyc])
  base <- switch(wt,
    MEC = d$WTMEC, INT = d$WTINT, SAF = d$WTSAF2YR,
    FOL = ifelse(d$cycle == "P", d$WTFOLPRP, d$WTMEC)
  )
  w <- ifelse(is.na(base), 0, base) * YRS[d$cycle] / Tt
  ifelse(w > 1e-6, w, 0)
}

## Harrell-style restricted cubic spline basis (matches Python rcs_basis)
rcs <- function(x, t) {
  k <- length(t)
  p <- function(u) pmax(u, 0)^3
  out <- cbind(x)
  for (j in 1:(k - 2)) {
    out <- cbind(
      out,
      (p(x - t[j]) -
         p(x - t[k - 1]) * (t[k] - t[j]) / (t[k] - t[k - 1]) +
         p(x - t[k]) * (t[k - 1] - t[j]) / (t[k] - t[k - 1])) /
        (t[k] - t[1])^2
    )
  }
  colnames(out) <- paste0("S", 0:(k - 2))
  out
}

IRON_CYC <- c("E", "F", "I", "P")
FORM <- ~ segf + age + age2 + race + educ + parity + smk + pircat + married + cyc

prep_iron <- function() {
  d <- D[cycle %in% IRON_CYC]
  d$w <- mkw(d, "MEC", IRON_CYC)
  d$y <- d$body_iron
  d$age <- d$RIDAGEYR - 32
  d$age2 <- d$age^2
  d$race <- relevel(factor(d$race), "NHW")
  d$educ <- relevel(factor(d$educ), ">HS")
  d$parity <- relevel(factor(d$parity), "0")
  d$smk <- relevel(factor(d$smk), "never")
  d$pircat <- relevel(factor(d$pircat), "1.3-3.5")
  d$cyc <- factor(d$cycle)
  d$segf <- relevel(factor(ifelse(d$grp == "pp", d$seg, "ctrl")), "ctrl")
  d$dom <- d$grp %in% c("pp", "ctrl") & d$w > 0 & !is.na(d$y)
  d$wd <- ifelse(d$w > 0, d$w, 1)
  des <- svydesign(ids = ~PSU, strata = ~STR, weights = ~wd, nest = TRUE, data = d)
  list(d = d, des = subset(des, dom))
}

## --- (1) three-segment contrasts ---
P <- prep_iron()
f_seg <- svyglm(update(FORM, y ~ .), design = P$des, family = gaussian())
cf <- coef(f_seg); V <- vcov(f_seg); dfd <- degf(P$des)
seg_rows <- lapply(c("0-6", "7-12", "13-24"), function(s) {
  nm <- paste0("segf", s)
  b <- unname(cf[nm]); se <- sqrt(V[nm, nm])
  tcrit <- qt(0.975, dfd)
  data.frame(
    model = "bodyiron_seg", term = paste0("seg", s),
    b = b, se = se, lo = b - tcrit * se, hi = b + tcrit * se,
    p = 2 * pt(-abs(b / se), dfd),
    n = nrow(P$des$variables), df_design = dfd,
    stringsAsFactors = FALSE
  )
})
seg_tab <- do.call(rbind, seg_rows)
rt <- regTermTest(f_seg, ~segf, method = "Wald", df = dfd)
seg_tab <- rbind(
  seg_tab,
  data.frame(model = "bodyiron_seg", term = "wald3_p",
             b = unname(rt$p), se = NA, lo = NA, hi = NA, p = unname(rt$p),
             n = nrow(P$des$variables), df_design = dfd,
             stringsAsFactors = FALSE)
)
fwrite(seg_tab, file.path(OUTD, "bodyiron_segments.csv"))
cat("Segment contrasts (mg/kg vs comparators):\n")
print(seg_tab, digits = 4)

## --- (2) RCS month grid + last significant month ---
d <- P$d
## knots = unweighted percentiles 5/35/65/95 among PP with weight>0 and non-missing y
kn <- as.numeric(quantile(d$month[d$grp == "pp" & d$w > 0 & !is.na(d$y)],
                          c(0.05, 0.35, 0.65, 0.95), type = 7))
cat("RCS knots (months):", kn, "\n")
Bm <- rcs(ifelse(is.na(d$month), 0, d$month), kn)
pp <- as.numeric(d$grp == "pp")
for (j in 1:ncol(Bm)) d[[paste0("S", j - 1)]] <- Bm[, j] * pp
d$PP <- pp
des <- subset(
  svydesign(ids = ~PSU, strata = ~STR, weights = ~wd, nest = TRUE, data = d),
  dom
)
f_rcs <- svyglm(
  y ~ PP + S0 + S1 + S2 + age + age2 + race + educ + parity + smk + pircat + married + cyc,
  design = des, family = gaussian()
)
grid <- 1:24
G <- rcs(grid, kn)
curve <- lapply(seq_along(grid), function(i) {
  L <- setNames(rep(0, length(coef(f_rcs))), names(coef(f_rcs)))
  L["PP"] <- 1
  L[c("S0", "S1", "S2")] <- G[i, ]
  e <- svycontrast(f_rcs, L)
  b <- as.numeric(coef(e)); se <- as.numeric(SE(e))
  tcrit <- qt(0.975, degf(des))
  lo <- b - tcrit * se; hi <- b + tcrit * se
  data.frame(
    month = grid[i], b = b, se = se, lo = lo, hi = hi,
    sig = (lo > 0) | (hi < 0),
    stringsAsFactors = FALSE
  )
})
curve_tab <- do.call(rbind, curve)
last_sig <- if (any(curve_tab$sig)) max(curve_tab$month[curve_tab$sig]) else 0L
ret <- NA_integer_
for (mm in 1:24) {
  if (!any(curve_tab$sig[mm:24])) { ret <- mm; break }
}
summary_row <- data.frame(
  outcome = "body_iron",
  knots = paste(round(kn), collapse = "/"),
  last_sig_month = last_sig,
  returns_to_ctrl_from_month = ret,
  n = nrow(des$variables),
  df_design = degf(des),
  stringsAsFactors = FALSE
)
fwrite(curve_tab, file.path(OUTD, "bodyiron_rcs_curve.csv"))
fwrite(summary_row, file.path(OUTD, "bodyiron_rcs_summary.csv"))
cat("Last significant month (CI excludes 0):", last_sig, "\n")
cat("Returns to comparator from month:", ret, "\n")
print(curve_tab[curve_tab$month %in% c(3, 6, 12, 14, 15, 18), ], digits = 4)
cat("Wrote:", file.path(OUTD, "bodyiron_segments.csv"), "\n")
cat("Wrote:", file.path(OUTD, "bodyiron_rcs_curve.csv"), "\n")
cat("Wrote:", file.path(OUTD, "bodyiron_rcs_summary.csv"), "\n")
