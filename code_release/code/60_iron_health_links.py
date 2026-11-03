"""Exploratory: postpartum iron stores vs health factors; cross-sectional discrimination; sample sizes.
Complex survey (svy.py). Does NOT modify manuscript. Outputs output/iron_links/*.csv
"""
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, './code')
import numpy as np, pandas as pd
from scipy import stats
import importlib
M = importlib.import_module('11_step1_models')
from svy import svyglm, svymean

O = './output/iron_links/'
os.makedirs(O, exist_ok=True)
IRONC = ['E', 'F', 'I', 'P']
A = M.A
tc = M.tcrit

# ---- helpers ----
def fdr_bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[ok] = q[np.argsort(order)]
    return out

def cov_base(d, cycles, include_month=True, include_seg=False, drop=()):
    X = M.covmat(d.fillna({'RIDAGEYR': 32}), cycles, pir='cat')
    for c in drop:
        if c in X.columns:
            X = X.drop(columns=[c])
    if include_month:
        X['month'] = d.month.fillna(0).astype(float)  # 0 for ctrl if present; PP-only uses real month
    if include_seg:
        for s in ['0-6', '7-12', '13-24']:
            X[f'seg={s}'] = (d.seg == s).astype(float)
    return X

def fit_domain(d, y, w, X, family='gaussian'):
    f = svyglm(X.fillna(0).values, y.values, w.values, d.STR.values, d.PSU.values,
               family=family, names=list(X.columns))
    return f

def coef_row(f, term, scale='abs', per=1.0, label='', n=None, family='gaussian'):
    if term not in f.names:
        return None
    b, se = f.coef(term)
    t = tc(f)
    lo, hi = b - t * se, b + t * se
    p = 2 * (1 - stats.t.cdf(abs(b / se), f.dfd))
    b2, lo2, hi2 = b * per, lo * per, hi * per
    if scale == 'pct' or (scale == 'log' and family == 'gaussian'):
        # percent change per `per` units of exposure (exposure already on log if ferritin)
        est = 100 * (np.exp(b2) - 1)
        loe = 100 * (np.exp(lo2) - 1)
        hie = 100 * (np.exp(hi2) - 1)
        sc = '% change'
    elif scale == 'OR' or family == 'binomial':
        est, loe, hie = np.exp(b2), np.exp(lo2), np.exp(hi2)
        sc = 'OR'
    else:
        est, loe, hie = b2, lo2, hi2
        sc = 'abs diff'
    return dict(label=label, term=term, n=n if n is not None else f.n, Nhat=f.N,
                est=est, lo=loe, hi=hie, p=p, scale=sc, df=f.dfd, b_raw=b, se_raw=se)

# ============================================================
# 1) Within postpartum: iron -> health outcomes
# ============================================================
# exposure: body_iron (per 1 SD in PP), log ferritin (per 1 SD of log), log sTfR (per 1 SD of log)
# adjust: age, race, educ, parity, smk, pircat, married, cycle, month since delivery
# fasting outcomes: SAF weight; sleep: INT; else MEC

OUTCOMES = [
    # key, label, transform, weight, family, min_n note
    ('BMXWAIST', 'Waist cm', None, 'MEC', 'gaussian'),
    ('BMXBMI', 'BMI kg/m2', None, 'MEC', 'gaussian'),
    ('sleep', 'Sleep h/night', None, 'INT', 'gaussian'),
    ('phq9', 'PHQ-9 score', None, 'MEC', 'gaussian'),
    ('LBXGH', 'HbA1c %', None, 'MEC', 'gaussian'),
    ('LBXGLU', 'FPG mg/dL (fasting)', None, 'SAF', 'gaussian'),
    ('LBXTC', 'Total cholesterol mg/dL', None, 'MEC', 'gaussian'),
    ('LBDHDD', 'HDL-C mg/dL', None, 'MEC', 'gaussian'),
    ('LBXTR', 'TG mg/dL (fasting)', 'log', 'SAF', 'gaussian'),
    ('LBDLDL', 'LDL-C mg/dL (fasting)', None, 'SAF', 'gaussian'),
    ('homa', 'HOMA-IR (fasting)', 'log', 'SAF', 'gaussian'),
    ('sbp', 'SBP mmHg', None, 'MEC', 'gaussian'),
    ('dbp', 'DBP mmHg', None, 'MEC', 'gaussian'),
    ('crp', 'CRP mg/L', 'log', 'MEC', 'gaussian'),
    ('LBXHGB', 'Hb g/dL', None, 'MEC', 'gaussian'),
]
BIN_OUT = [
    ('short_sleep', 'Sleep <7 h', 'sleep', lambda v: (v < 7).astype(float), 'INT'),
    ('phq10', 'PHQ-9 >=10', 'phq9', lambda v: (v >= 10).astype(float), 'MEC'),
    ('waist88', 'Waist >=88 cm', 'BMXWAIST', lambda v: (v >= 88).astype(float), 'MEC'),
    ('anemia', 'Anemia Hb<12', 'LBXHGB', lambda v: (v < 12).astype(float), 'MEC'),
    ('a1c_pre', 'HbA1c >=5.7%', 'LBXGH', lambda v: (v >= 5.7).astype(float), 'MEC'),
]

# Prepare PP iron domain base
base = A[A.cycle.isin(IRONC)].copy()
# keep design dummies
assert (base.grp == 'dummy').any() or True

def pp_mask(d):
    return d.grp == 'pp'

def ctrl_mask(d):
    return d.grp == 'ctrl'

# SD of exposures in PP with body_iron
pp_bi = base.loc[(base.grp == 'pp') & base.body_iron.notna(), 'body_iron']
SD_BI = float(pp_bi.std(ddof=1))
pp_lf = np.log(base.loc[(base.grp == 'pp') & base.ferritin.gt(0), 'ferritin'])
SD_LF = float(pp_lf.std(ddof=1))
pp_ls = np.log(base.loc[(base.grp == 'pp') & base.stfr.gt(0), 'stfr'])
SD_LS = float(pp_ls.std(ddof=1))
print(f'SD body_iron PP={SD_BI:.3f}; SD log ferritin={SD_LF:.3f}; SD log sTfR={SD_LS:.3f}')

EXPOSURES = [
    ('body_iron', None, 'Body iron (per 1 SD)', SD_BI),
    ('ferritin', 'log', 'log Ferritin (per 1 SD)', SD_LF),
    ('stfr', 'log', 'log sTfR (per 1 SD)', SD_LS),
]

rows_assoc = []
# Primary family for FDR: body_iron -> continuous health (PP only)
fam_keys = []

for domain_name, dmask in [('postpartum', pp_mask), ('control', ctrl_mask)]:
    for exp_key, exp_tr, exp_lab, sd in EXPOSURES:
        for ykey, ylab, ytr, wt, fam in OUTCOMES:
            d = base.copy()
            # y
            y = d[ykey].astype(float)
            if ytr == 'log':
                y = np.log(y.where(y > 0))
            # exposure
            xexp = d[exp_key].astype(float)
            if exp_tr == 'log':
                xexp = np.log(xexp.where(xexp > 0))
            w = M.W(d, wt, IRONC) * dmask(d).astype(float)
            w = w * np.isfinite(y).astype(float) * np.isfinite(xexp).astype(float)
            n = int((w > 0).sum())
            # skip if too small
            if n < 80:
                rows_assoc.append(dict(
                    domain=domain_name, exposure=exp_lab, outcome=ylab, n=n,
                    est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, scale='skipped n<80',
                    note='sample too small', family_fdr='skip'))
                continue
            # for control domain, no month; for PP include month
            X = cov_base(d, IRONC, include_month=(domain_name == 'postpartum'), include_seg=False)
            # drop parity=0 rare in PP? keep as is
            X['exposure'] = xexp.fillna(0).values
            # zero weight already handles missings
            f = fit_domain(d, y, w, X, family='gaussian')
            scale = 'pct' if ytr == 'log' else 'abs'
            r = coef_row(f, 'exposure', scale=scale, per=sd, label=f'{exp_lab} -> {ylab}', n=n)
            if r is None:
                continue
            r.update(dict(domain=domain_name, exposure=exp_lab, outcome=ylab,
                          n_outcome=n, note='', family_fdr=f'{domain_name}|{exp_key}|cont'))
            rows_assoc.append(r)

        for bkey, blab, src, fn, wt in BIN_OUT:
            d = base.copy()
            y = fn(d[src].astype(float))
            xexp = d[exp_key].astype(float)
            if exp_tr == 'log':
                xexp = np.log(xexp.where(xexp > 0))
            w = M.W(d, wt, IRONC) * dmask(d).astype(float)
            w = w * y.notna().astype(float) * np.isfinite(xexp).astype(float)
            n = int((w > 0).sum())
            n_ev = int(((y == 1) & (w > 0)).sum())
            if n < 80 or n_ev < 15:
                rows_assoc.append(dict(
                    domain=domain_name, exposure=exp_lab, outcome=blab, n=n, n_events=n_ev,
                    est=np.nan, lo=np.nan, hi=np.nan, p=np.nan, scale='skipped',
                    note=f'n={n}, events={n_ev} too small', family_fdr='skip'))
                continue
            X = cov_base(d, IRONC, include_month=(domain_name == 'postpartum'))
            X['exposure'] = xexp.fillna(0).values
            f = fit_domain(d, y.astype(float), w, X, family='binomial')
            r = coef_row(f, 'exposure', scale='OR', per=sd, label=f'{exp_lab} -> {blab}', n=n, family='binomial')
            r.update(dict(domain=domain_name, exposure=exp_lab, outcome=blab, n_events=n_ev,
                          note='', family_fdr=f'{domain_name}|{exp_key}|bin'))
            rows_assoc.append(r)

T1 = pd.DataFrame(rows_assoc)
# FDR within families that are not skip
T1['q'] = np.nan
for fam, g in T1.groupby('family_fdr'):
    if fam == 'skip':
        continue
    T1.loc[g.index, 'q'] = fdr_bh(g['p'].values)

T1.to_csv(O + 'T1_iron_to_health_adjusted.csv', index=False)
print('T1 rows', len(T1), 'PP body_iron cont sig q<0.05:',
      ((T1.domain == 'postpartum') & (T1.exposure.str.startswith('Body iron')) &
       (T1.family_fdr.str.endswith('cont')) & (T1.q < 0.05)).sum())

# ============================================================
# 2) Demographics / BF / parity predicting body iron (PP)
# ============================================================
# Model body_iron ~ factor (+ age, cycle, month; for SES models add race etc carefully)
# Report adjusted mean differences

demo_rows = []
d = base.copy()
y = d['body_iron'].astype(float)
w0 = M.W(d, 'MEC', IRONC) * (d.grp == 'pp').astype(float) * np.isfinite(y).astype(float)

# 2a race (ref NHW)
X = pd.DataFrame(index=d.index)
X['const'] = 1.0
a = d.RIDAGEYR.fillna(32) - 32
X['age'] = a; X['age2'] = a ** 2
X['month'] = d.month.fillna(0)
for c in IRONC[1:]:
    X[f'cyc={c}'] = (d.cycle == c).astype(float)
for lev in ['MexAm', 'OthHisp', 'NHB', 'Other']:
    X[f'race={lev}'] = (d.race == lev).astype(float)
w = w0 * d.race.notna().astype(float)
f = fit_domain(d, y, w, X)
for lev in ['MexAm', 'OthHisp', 'NHB', 'Other']:
    r = coef_row(f, f'race={lev}', per=1.0, label=f'race {lev} vs NHW', n=int((w > 0).sum()))
    r.update(dict(factor='race', level=lev, ref='NHW', family_fdr='demo_race'))
    demo_rows.append(r)

# 2b PIR categories (ref 1.3-3.5); keep miss as category
X = pd.DataFrame(index=d.index)
X['const'] = 1.0
X['age'] = a; X['age2'] = a ** 2; X['month'] = d.month.fillna(0)
for c in IRONC[1:]:
    X[f'cyc={c}'] = (d.cycle == c).astype(float)
for lev in ['<1.3', '>=3.5', 'miss']:
    X[f'pir={lev}'] = (d.pircat == lev).astype(float)
# also adjust race
for lev in ['MexAm', 'OthHisp', 'NHB', 'Other']:
    X[f'race={lev}'] = (d.race == lev).astype(float)
w = w0.copy()
f = fit_domain(d, y, w, X)
for lev in ['<1.3', '>=3.5', 'miss']:
    r = coef_row(f, f'pir={lev}', per=1.0, label=f'PIR {lev} vs 1.3-3.5', n=int((w > 0).sum()))
    r.update(dict(factor='PIR', level=lev, ref='1.3-3.5', family_fdr='demo_pir'))
    demo_rows.append(r)

# 2c parity (ref 1) among PP (parity 0 rare)
X = pd.DataFrame(index=d.index)
X['const'] = 1.0
X['age'] = a; X['age2'] = a ** 2; X['month'] = d.month.fillna(0)
for c in IRONC[1:]:
    X[f'cyc={c}'] = (d.cycle == c).astype(float)
for lev in ['MexAm', 'OthHisp', 'NHB', 'Other']:
    X[f'race={lev}'] = (d.race == lev).astype(float)
for lev in ['<1.3', '>=3.5', 'miss']:
    X[f'pir={lev}'] = (d.pircat == lev).astype(float)
for lev in ['0', '2', '3+']:
    X[f'parity={lev}'] = (d.parity == lev).astype(float)
w = w0 * d.parity.notna().astype(float)
f = fit_domain(d, y, w, X)
for lev in ['0', '2', '3+']:
    if f'parity={lev}' not in f.names:
        continue
    r = coef_row(f, f'parity={lev}', per=1.0, label=f'parity {lev} vs 1', n=int((w > 0).sum()))
    r.update(dict(factor='parity', level=lev, ref='1', family_fdr='demo_parity'))
    demo_rows.append(r)

# 2d breastfeeding 0-12 mo only (BF vs noBF), adjust age race pir month cycle
d12 = base.copy()
y12 = d12['body_iron'].astype(float)
w12 = M.W(d12, 'MEC', IRONC) * ((d12.grp == 'pp') & d12.seg.isin(['0-6', '7-12']) & d12.bf.notna()).astype(float)
w12 = w12 * np.isfinite(y12).astype(float)
X = pd.DataFrame(index=d12.index)
X['const'] = 1.0
a12 = d12.RIDAGEYR.fillna(32) - 32
X['age'] = a12; X['age2'] = a12 ** 2
X['month'] = d12.month.fillna(0)
X['bf'] = d12.bf.fillna(0).astype(float)
for c in IRONC[1:]:
    X[f'cyc={c}'] = (d12.cycle == c).astype(float)
for lev in ['MexAm', 'OthHisp', 'NHB', 'Other']:
    X[f'race={lev}'] = (d12.race == lev).astype(float)
for lev in ['<1.3', '>=3.5', 'miss']:
    X[f'pir={lev}'] = (d12.pircat == lev).astype(float)
f = fit_domain(d12, y12, w12, X)
r = coef_row(f, 'bf', per=1.0, label='currently breastfeeding vs not (0-12 mo)', n=int((w12 > 0).sum()))
r.update(dict(factor='breastfeeding', level='BF', ref='noBF', family_fdr='demo_bf',
              n_bf=int(((d12.bf == 1) & (w12 > 0)).sum()),
              n_nobf=int(((d12.bf == 0) & (w12 > 0)).sum())))
demo_rows.append(r)

# 2e CRP (log) association with body iron — inflammation pathway (also in T1 reverse)
# already in T1 as iron->crp; also report CRP->iron for completeness
X = cov_base(base, IRONC, include_month=True)
lcrp = np.log(base.crp.where(base.crp > 0))
X['log_crp'] = lcrp.fillna(0)
w = M.W(base, 'MEC', IRONC) * (base.grp == 'pp').astype(float) * np.isfinite(y).astype(float) * np.isfinite(lcrp).astype(float)
f = fit_domain(base, y, w, X)
# per doubling of CRP = log(2)
r = coef_row(f, 'log_crp', per=np.log(2), label='body iron per CRP doubling', n=int((w > 0).sum()))
r.update(dict(factor='CRP', level='per doubling', ref='', family_fdr='demo_crp'))
demo_rows.append(r)

T2 = pd.DataFrame(demo_rows)
T2['q'] = np.nan
for fam, g in T2.groupby('family_fdr'):
    T2.loc[g.index, 'q'] = fdr_bh(g['p'].values)
T2.to_csv(O + 'T2_factors_to_body_iron_PP.csv', index=False)

# ============================================================
# 3) Cross-sectional discrimination (NOT temporal prediction)
# ============================================================
# Unweighted ROC AUC on PP with body_iron (honestly note: design weights not in AUC;
# also report weighted logistic OR which is already in T1)
from sklearn.metrics import roc_auc_score

disc_rows = []
pp = base[(base.grp == 'pp') & base.body_iron.notna()].copy()

def auc_safe(y, score):
    y = np.asarray(y); score = np.asarray(score)
    m = np.isfinite(y) & np.isfinite(score)
    y, score = y[m], score[m]
    if y.sum() < 10 or (1 - y).sum() < 10:
        return np.nan, int(m.sum()), int(y.sum())
    return float(roc_auc_score(y, score)), int(m.sum()), int(y.sum())

targets = [
    ('anemia', (pp.LBXHGB < 12).astype(float), 'Anemia Hb<12'),
    ('short_sleep', (pp.sleep < 7).astype(float), 'Sleep <7 h'),
    ('phq10', (pp.phq9 >= 10).astype(float), 'PHQ-9 >=10'),
    ('waist88', (pp.BMXWAIST >= 88).astype(float), 'Waist >=88 cm'),
    ('a1c_pre', (pp.LBXGH >= 5.7).astype(float), 'HbA1c >=5.7%'),
    ('bi_neg', (pp.body_iron < 0).astype(float), 'Body iron <0 (sanity)'),
]
# MetS (fasting subsample): need waist, TG, HDL, BP, FPG — at least 3 of 5
w_saf = M.W(base, 'SAF', IRONC)
pp_f = base[(base.grp == 'pp') & base.body_iron.notna() & (w_saf.reindex(base.index).fillna(0) > 0)].copy()
comp = pd.DataFrame({
    'waist': (pp_f.BMXWAIST >= 88).astype(float),
    'tg': (pp_f.LBXTR >= 150).astype(float),
    'hdl': (pp_f.LBDHDD < 50).astype(float),
    'bp': ((pp_f.sbp >= 130) | (pp_f.dbp >= 85)).astype(float),
    'fpg': (pp_f.LBXGLU >= 100).astype(float),
})
# only count components with non-missing
valid = pp_f.BMXWAIST.notna().astype(int) + pp_f.LBXTR.notna().astype(int) + pp_f.LBDHDD.notna().astype(int) + \
        (pp_f.sbp.notna() | pp_f.dbp.notna()).astype(int) + pp_f.LBXGLU.notna().astype(int)
n_comp = comp.sum(axis=1)
mets = (n_comp >= 3).astype(float).where(valid >= 4)  # require >=4 components assessed

for key, ybin, lab in targets:
    # higher body iron should lower anemia risk → score = -body_iron for anemia/bi_neg; for others try both
    for direction, score in [('higher_iron', pp.body_iron.values),
                             ('lower_iron', -pp.body_iron.values)]:
        auc, n, nev = auc_safe(ybin.values, score)
        disc_rows.append(dict(outcome=lab, score=f'body_iron ({direction})', auc=auc, n=n, n_events=nev, sample='PP iron cycles'))
    # ferritin
    for direction, score in [('higher_ferr', pp.ferritin.values),
                             ('lower_ferr', -pp.ferritin.values)]:
        auc, n, nev = auc_safe(ybin.values, score)
        disc_rows.append(dict(outcome=lab, score=f'ferritin ({direction})', auc=auc, n=n, n_events=nev, sample='PP iron cycles'))

# MetS
if mets.notna().sum() >= 80:
    for direction, score in [('higher_iron', pp_f.body_iron.values),
                             ('lower_iron', -pp_f.body_iron.values)]:
        auc, n, nev = auc_safe(mets.values, score)
        disc_rows.append(dict(outcome='MetS (>=3/5 components)', score=f'body_iron ({direction})',
                              auc=auc, n=n, n_events=nev, sample='PP fasting iron cycles'))

# Incremental value: age-only vs age+body_iron logistic predicted probabilities (unweighted AUC)
from sklearn.linear_model import LogisticRegression

def auc_incremental(y, age, iron):
    m = np.isfinite(y) & np.isfinite(age) & np.isfinite(iron)
    y, age, iron = y[m], age[m], iron[m]
    if y.sum() < 15 or (1 - y).sum() < 15:
        return np.nan, np.nan, int(m.sum()), int(y.sum())
    lr0 = LogisticRegression(max_iter=500)
    lr1 = LogisticRegression(max_iter=500)
    X0 = age.reshape(-1, 1)
    X1 = np.column_stack([age, iron])
    p0 = lr0.fit(X0, y).predict_proba(X0)[:, 1]
    p1 = lr1.fit(X1, y).predict_proba(X1)[:, 1]
    return float(roc_auc_score(y, p0)), float(roc_auc_score(y, p1)), int(m.sum()), int(y.sum())

inc_rows = []
for key, ybin, lab in targets:
    if key == 'bi_neg':
        continue
    a0, a1, n, nev = auc_incremental(ybin.values, pp.RIDAGEYR.values, pp.body_iron.values)
    inc_rows.append(dict(outcome=lab, auc_age_only=a0, auc_age_plus_body_iron=a1,
                         delta=a1 - a0 if np.isfinite(a0) and np.isfinite(a1) else np.nan,
                         n=n, n_events=nev, note='unweighted in-sample AUC; optimistic; NOT temporal prediction'))

if mets.notna().sum() >= 80:
    a0, a1, n, nev = auc_incremental(mets.values, pp_f.RIDAGEYR.values, pp_f.body_iron.values)
    inc_rows.append(dict(outcome='MetS (>=3/5)', auc_age_only=a0, auc_age_plus_body_iron=a1,
                         delta=a1 - a0 if np.isfinite(a0) and np.isfinite(a1) else np.nan,
                         n=n, n_events=nev, note='unweighted fasting subsample; NOT temporal prediction'))

T3a = pd.DataFrame(disc_rows)
T3b = pd.DataFrame(inc_rows)
T3a.to_csv(O + 'T3a_discrimination_AUC_unweighted.csv', index=False)
T3b.to_csv(O + 'T3b_incremental_AUC_age_vs_age_iron.csv', index=False)

# ============================================================
# 4) Sample size / coverage table
# ============================================================
cov = []
pp_all = base[base.grp == 'pp']
for label, mask in [
    ('body_iron', pp_all.body_iron.notna()),
    ('+ waist', pp_all.body_iron.notna() & pp_all.BMXWAIST.notna()),
    ('+ BMI', pp_all.body_iron.notna() & pp_all.BMXBMI.notna()),
    ('+ sleep', pp_all.body_iron.notna() & pp_all.sleep.notna()),
    ('+ PHQ-9', pp_all.body_iron.notna() & pp_all.phq9.notna()),
    ('+ HbA1c', pp_all.body_iron.notna() & pp_all.LBXGH.notna()),
    ('+ CRP', pp_all.body_iron.notna() & pp_all.crp.notna()),
    ('+ SBP', pp_all.body_iron.notna() & pp_all.sbp.notna()),
    ('+ breastfeeding known', pp_all.body_iron.notna() & pp_all.bf.notna()),
    ('+ fasting FPG (SAF>0)', pp_all.body_iron.notna() & (M.W(base, 'SAF', IRONC).reindex(pp_all.index).fillna(0).values > 0) & pp_all.LBXGLU.notna()),
    ('+ fasting TG (SAF>0)', pp_all.body_iron.notna() & (M.W(base, 'SAF', IRONC).reindex(pp_all.index).fillna(0).values > 0) & pp_all.LBXTR.notna()),
]:
    cov.append(dict(item=label, n=int(mask.sum())))
pd.DataFrame(cov).to_csv(O + 'T0_coverage_PP_iron.csv', index=False)

# Weighted means of body iron by tertile of waist etc for description
desc = []
pp2 = pp.copy()
pp2['waist_tert'] = pd.qcut(pp2.BMXWAIST, 3, labels=['T1 low', 'T2', 'T3 high'])
wpp = M.W(base, 'MEC', IRONC).reindex(pp2.index)
for t, g in pp2.groupby('waist_tert'):
    ww = wpp.loc[g.index].values
    mu, se, n = svymean(g.body_iron.values, ww, base.loc[g.index, 'STR'].values, base.loc[g.index, 'PSU'].values)
    desc.append(dict(by='waist tertile', level=str(t), mean_body_iron=mu, se=se, n=n))
pd.DataFrame(desc).to_csv(O + 'T4_desc_body_iron_by_waist_tertile.csv', index=False)

# Print key findings for doc writing
print('\n=== PP body_iron -> health (q sorted) ===')
sub = T1[(T1.domain == 'postpartum') & (T1.exposure.str.startswith('Body iron')) & (T1.family_fdr != 'skip')].copy()
sub = sub.sort_values('p')
print(sub[['outcome', 'est', 'lo', 'hi', 'p', 'q', 'scale', 'n']].head(20).to_string(index=False))
print('\n=== Demo -> body_iron ===')
print(T2[['factor', 'level', 'est', 'lo', 'hi', 'p', 'q']].to_string(index=False))
print('\n=== Incremental AUC ===')
print(T3b.to_string(index=False))
print('\nDone. SD_BI=', SD_BI)
