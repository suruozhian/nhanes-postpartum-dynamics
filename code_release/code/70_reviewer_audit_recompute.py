"""Reviewer/statistical audit recomputations (2026-10-30). Read-only w.r.t. manuscript; writes output/audit/*.csv"""
import sys,importlib.util,numpy as np,pandas as pd
B='./'; sys.path.insert(0,B+'code')
spec=importlib.util.spec_from_file_location('m',B+'code/11_step1_models.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from svy import svyglm, svymean
A0=pd.read_pickle(B+'data/step1_analytic.pkl')
out=[]
# 0. undisclosed recoding of missing covariates (pp+ctrl, MAIN cycles)
S=A0[A0.grp.isin(['pp','ctrl'])&A0.cycle.isin(m.MAIN)]
for v in ['smk','parity','educ','pircat']: out.append(('missing_count',v,int((S[v]=='miss').sum()),''))
mm=S.DMDMARTL.isna()&S.DMDMARTZ.isna(); out.append(('missing_count','marital_missing->not married',int(mm.sum()),''))
out.append(('missing_count','RIDEXPRG==3 (uncertain) retained in pp',int((S.pp&(S.RIDEXPRG==3)).sum()),''))
# 1. 25(OH)D adjusted for exam season (RIDEXMON)
mon={}
for c,f in [('E','DEMO_E'),('F','DEMO_F'),('G','DEMO_G'),('H','DEMO_H'),('I','DEMO_I'),('J','DEMO_J'),('P','P_DEMO')]:
    try: x=pd.read_sas(B+'data/raw/'+f+'.xpt',format='xport')[['SEQN','RIDEXMON']]; mon.update(dict(zip(x.SEQN,x.RIDEXMON)))
    except Exception as e: print('no',f,e)
A=m.A; A['nov_apr']=A.SEQN.map(mon).eq(1).astype(float)
def seg(key,cyc,wt,tr,extra_cols=(),mask=None):
    d,y,w,X=m.prep(key,cyc,wt,tr,extra_mask=mask)
    for c in extra_cols: X[c]=d[c].fillna(0).values
    for s in ['0-6','7-12','13-24']: X[f'seg={s}']=(d.seg==s).astype(float)
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    r=[]
    for s in ['0-6','7-12','13-24']:
        b,se=f.coef(f'seg={s}'); e,lo,hi=m.eff(b,se,f,tr); p=2*(1-m.stats.t.cdf(abs(b/se),f.dfd)); r.append((s,e,lo,hi,p))
    return r
V=['E','F','G','H','I','J']
for lab,cols in [('25OHD_base',()),('25OHD_+season',('nov_apr',))]:
    for s,e,lo,hi,p in seg('LBXVIDMS',V,'MEC',None,cols): out.append((lab,s,round(e,2),f'({lo:.2f},{hi:.2f}) p={p:.4f}'))
# season distribution by segment
d=A[A.cycle.isin(V)&A.grp.isin(['pp','ctrl'])&A.LBXVIDMS.notna()]
for g,sub in d.groupby('seg'): out.append(('pct_exam_Nov-Apr',g,round(100*sub.nov_apr.mean(),1),f'n={len(sub)}'))
# 2. comparators restricted to parous women (exclude never-pregnant)
par=lambda d: ~((d.grp=='ctrl')&(d.RHQ131==2))
for key,cyc,tr in [('body_iron',['E','F','I','P'],None),('ferritin',['E','F','I','P'],'log'),('rbc_folate',m.MAIN,'log'),('BMXWAIST',m.MAIN,None),('LBXSUA',m.MAIN,None)]:
    for s,e,lo,hi,p in seg(key,cyc,'MEC' if key!='rbc_folate' else 'FOL',tr,mask=par): out.append((f'parous_ctrl_only:{key}',s,round(e,2),f'({lo:.2f},{hi:.2f}) p={p:.4f}'))
# 3. knot sensitivity for last significant month (body iron, Hb, AST, HDL)
def lastsig(key,cyc,wt,tr,knots):
    d,y,w,X=m.prep(key,cyc,wt,tr); mth=d.month.where(d.grp=='pp')
    Bm=m.rcs_basis(mth.fillna(0).values,knots); pp=(d.grp=='pp').astype(float).values; X['PP']=pp
    for j in range(Bm.shape[1]): X[f'S{j}']=Bm[:,j]*pp
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    G=m.rcs_basis(np.arange(1,25),knots); L=np.zeros((24,len(f.b))); L[:,f.names.index('PP')]=1
    for j in range(Bm.shape[1]): L[:,f.names.index(f'S{j}')]=G[:,j]
    est,se=f.contrast(L); t=m.stats.t.ppf(.975,f.dfd); sig=(est-t*se>0)|(est+t*se<0)
    s22=[i+1 for i in range(22) if sig[i]]
    return max([i+1 for i in range(24) if sig[i]],default=0), max(s22,default=0), ''.join('x' if v else '.' for v in sig)
for key,cyc,wt,tr in [('body_iron',['E','F','I','P'],'MEC',None),('LBXHGB',m.MAIN,'MEC',None),('LBXSASSI',m.MAIN,'MEC','log'),('LBDHDD',m.MAIN,'MEC',None),('rbc_folate',m.MAIN,'FOL','log')]:
    for kn in [[1,7,13,22],[3,12,21],[1,5,10,16,22],[2,6,12,18,23]]:
        a,b,pat=lastsig(key,cyc,wt,tr,kn); out.append((f'knots:{key}','/'.join(map(str,kn)),a,f'last_sig_within_1-22={b}; months1-24={pat}'))
# 4. effective sample size / RSE for small prevalence cells (NCHS-style reliability)
d=A[A.cycle.isin(['E','F','I','P'])&A.grp.isin(['pp','ctrl'])]; w=m.W(d,'MEC',['E','F','I','P'])
for lab,yb in [('body_iron<0',(d.body_iron<0).astype(float).where(d.body_iron.notna())),('IDA',((d.body_iron<0)&(d.LBXHGB<12)).astype(float).where(d.body_iron.notna()&d.LBXHGB.notna()))]:
    for s in ['0-6','7-12','13-24','ctrl']:
        ww=(w*(d.seg==s)*yb.notna()).values; p,se,n=svymean(yb.fillna(0).values,ww,d.STR.values,d.PSU.values)
        neff=p*(1-p)/se**2 if se>0 else np.nan; ncase=int(((yb==1)&(ww>0)).sum())
        out.append((f'reliability:{lab}',s,round(100*p,1),f'n={n} cases={ncase} SE%={100*se:.1f} RSE={se/p if p>0 else np.nan:.2f} n_eff={neff:.0f} flag={(neff<30) or (ncase<10) or (se/p>0.3 if p>0 else True)}'))
pd.DataFrame(out,columns=['check','level','value','detail']).to_csv(B+'output/audit/reviewer_audit_recompute.csv',index=False)
print(pd.DataFrame(out).to_string())
