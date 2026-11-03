"""Raw log-scale coefficients/SE from svy.py for the R cross-check models."""
import sys,importlib; sys.path.insert(0,'./code')
M=importlib.import_module('11_step1_models'); from svy import svyglm, svymean, rcs_basis
import numpy as np, pandas as pd
A=M.A; R=[]
def segfit(tag,key,cyc,wt,tr,family='gaussian',ybin=None,mask=None):
    d,y,w,X=M.prep(key,cyc,wt,tr if ybin is None else None,extra_mask=mask)
    if ybin is not None: y=ybin(d[key]).astype(float).where(d[key].notna())
    for s in ['0-6','7-12','13-24']: X[f'seg={s}']=(d.seg==s).astype(float)
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,family,list(X.columns))
    for s in ['0-6','7-12','13-24']:
        b,se=f.coef(f'seg={s}'); R.append(dict(model=tag,term=f'seg{s}',b=b,se=se,n=f.n,sumw=f.N,df=f.dfd))
    R.append(dict(model=tag,term='wald3_p',b=f.wald([f'seg={s}' for s in ['0-6','7-12','13-24']])[2],se=np.nan,n=f.n,sumw=f.N,df=f.dfd))
segfit('ferritin_log','ferritin',['E','F','I','P'],'MEC','log')
segfit('homa_log','homa',M.MAIN,'SAF','log')
segfit('rbcfol_log','rbc_folate',M.MAIN,'FOL','log')
segfit('bodyiron','body_iron',['E','F','I','P'],'MEC',None)
segfit('shortsleep_logit','sleep',M.MAIN,'INT',None,'binomial',lambda v:(v<7))
# RCS (ferritin, 4 knots) contrasts at months 3,6,12,18
d,y,w,X=M.prep('ferritin',['E','F','I','P'],'MEC','log'); m=d.month.where(d.grp=='pp')
kn=np.nanpercentile(m[w>0],[5,35,65,95]); Bm=rcs_basis(m.fillna(0).values,kn); pp=(d.grp=='pp').astype(float).values
X['PP']=pp
for j in range(Bm.shape[1]): X[f'S{j}']=Bm[:,j]*pp
f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
G=rcs_basis(np.array([3,6,12,18]),kn); L=np.zeros((4,len(f.b))); L[:,f.names.index('PP')]=1
for j in range(Bm.shape[1]): L[:,f.names.index(f'S{j}')]=G[:,j]
e,se=f.contrast(L)
for mm,a,b in zip([3,6,12,18],e,se): R.append(dict(model='ferritin_rcs',term=f'm{mm}',b=a,se=b,n=f.n,sumw=f.N,df=f.dfd))
print('knots',kn)
# weighted mean ferritin 0-6
d,y,w,X=M.prep('ferritin',['E','F','I','P'],'MEC',None); ww=(w*(d.seg=='0-6')).values
mu,se,n=svymean(y.values,ww,d.STR.values,d.PSU.values); R.append(dict(model='ferritin_mean_0-6',term='mean',b=mu,se=se,n=n,sumw=ww.sum(),df=np.nan))
pd.DataFrame(R).to_csv('./output/rcheck_py.csv',index=False); print(pd.DataFrame(R).to_string())
