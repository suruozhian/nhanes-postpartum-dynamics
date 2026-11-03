"""Minimal complex-survey regression (Taylor linearisation), equivalent in spirit to R survey::svyglm.
- Stratified, with-replacement PSU sandwich variance: V = A^-1 [sum_h n_h/(n_h-1) sum_j (z_hj - zbar_h)(z_hj - zbar_h)'] A^-1
- Domain (subset) analysis: rows outside the domain are KEPT with weight 0, so all PSUs/strata of the design remain
  (same as svydesign + subset()). Design df = (#PSU - #strata) over strata/PSUs that are in the design file.
- Linear (WLS) and logistic (IRLS) models.
"""
import numpy as np, pandas as pd
from scipy import stats

def _meat(U, strata, psu):
    df = pd.DataFrame(U); df['h'] = strata; df['j'] = psu
    Z = df.groupby(['h', 'j']).sum()
    M = np.zeros((U.shape[1], U.shape[1]))
    for h, g in Z.groupby(level=0):
        n = len(g)
        if n < 2: continue  # singleton stratum (should not occur in NHANES MVU)
        z = g.values; zc = z - z.mean(0)
        M += n / (n - 1) * zc.T @ zc
    return M

def design_df(strata, psu, w=None):
    d = pd.DataFrame({'h': strata, 'j': psu})
    return d.drop_duplicates().shape[0] - d.h.nunique()

class Fit:
    def __init__(self, names, b, V, dfd, n, N):
        self.names = list(names); self.b = b; self.V = V; self.dfd = dfd; self.n = n; self.N = N
    def coef(self, name):
        i = self.names.index(name); return self.b[i], np.sqrt(self.V[i, i])
    def contrast(self, L):
        L = np.atleast_2d(L); est = L @ self.b; se = np.sqrt(np.diag(L @ self.V @ L.T)); return est, se
    def wald(self, names=None, L=None):
        if L is None:
            idx = [self.names.index(n) for n in names]
            L = np.zeros((len(idx), len(self.b)));
            for r, i in enumerate(idx): L[r, i] = 1
        L = np.atleast_2d(L); q = L.shape[0]; e = L @ self.b
        X2 = float(e @ np.linalg.pinv(L @ self.V @ L.T) @ e)
        F = X2 / q; p = 1 - stats.f.cdf(F, q, max(self.dfd - q + 1, 1))
        return F, q, p
    def table(self, link='identity'):
        se = np.sqrt(np.diag(self.V)); t = stats.t.ppf(0.975, self.dfd)
        out = pd.DataFrame({'term': self.names, 'b': self.b, 'se': se, 'lo': self.b - t * se, 'hi': self.b + t * se,
                            'p': 2 * (1 - stats.t.cdf(np.abs(self.b / se), self.dfd))})
        return out

def svyglm(X, y, w, strata, psu, family='gaussian', names=None, maxit=50):
    """X: (N,p) incl. intercept; rows with w==0 or nan y are out-of-domain but kept for variance structure."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float).copy()
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1) & (w > 0)
    w[~ok] = 0; y = np.where(ok, y, 0.0); X = np.where(np.isfinite(X), X, 0.0)
    dfd = design_df(strata, psu)
    if family == 'gaussian':
        A = (X * w[:, None]).T @ X
        Ai = np.linalg.pinv(A); b = Ai @ ((X * w[:, None]).T @ y)
        e = y - X @ b; U = X * (w * e)[:, None]
    else:  # logistic IRLS
        b = np.zeros(X.shape[1])
        for _ in range(maxit):
            mu = 1 / (1 + np.exp(-(X @ b))); W = w * mu * (1 - mu)
            A = (X * W[:, None]).T @ X; Ai = np.linalg.pinv(A)
            step = Ai @ ((X * w[:, None]).T @ (y - mu)); b = b + step
            if np.max(np.abs(step)) < 1e-8: break
        mu = 1 / (1 + np.exp(-(X @ b))); A = (X * (w * mu * (1 - mu))[:, None]).T @ X; Ai = np.linalg.pinv(A)
        U = X * (w * (y - mu))[:, None]
    V = Ai @ _meat(U, strata, psu) @ Ai
    return Fit(names if names is not None else [f'x{i}' for i in range(X.shape[1])], b, V, dfd, int(ok.sum()), float(w.sum()))

def svymean(y, w, strata, psu):
    """weighted mean with linearised SE (domain = w>0 & finite y)"""
    y = np.asarray(y, float); w = np.asarray(w, float).copy(); ok = np.isfinite(y) & (w > 0); w[~ok] = 0; y = np.where(ok, y, 0)
    N = w.sum(); m = (w * y).sum() / N; U = (w * (y - m) / N)[:, None]
    se = np.sqrt(_meat(U, strata, psu)[0, 0]); return m, se, int(ok.sum())

def wquantile(y, w, q=0.5):
    y = np.asarray(y, float); w = np.asarray(w, float); ok = np.isfinite(y) & (w > 0); y, w = y[ok], w[ok]
    o = np.argsort(y); y, w = y[o], w[o]; c = np.cumsum(w) / w.sum(); return float(y[np.searchsorted(c, q)])

def rcs_basis(x, knots):
    """Harrell restricted cubic spline basis (linear term + k-2 nonlinear terms)."""
    x = np.asarray(x, float); t = np.asarray(knots, float); k = len(t); cols = [x]
    p = lambda u: np.where(u > 0, u, 0) ** 3
    for j in range(k - 2):
        cols.append((p(x - t[j]) - p(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                     + p(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)
