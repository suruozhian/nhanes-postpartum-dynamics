"""Revision R1-R8 recomputations (2026-10-30, user-approved). Design unchanged: domain via zero weights, all PSUs kept,
multi-cycle weights per NCHS, weights <=1e-6 set to 0. Outputs: output/revision/*.csv"""
import sys,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
B='./'; sys.path.insert(0,B+'code'); O=B+'output/revision/'
spec=importlib.util.spec_from_file_location('m',B+'code/11_step1_models.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from svy import svyglm, svymean
from scipy import stats
A=m.A; MAXM=22; GRID=np.arange(1,MAXM+1); rng=np.random.default_rng(20261030)
SEGS=['0-6','7-12','13-24']; IRONC=['E','F','I','P']
# extra derived vars (same definitions as 30_deep_dive.py)
A['bi_neg']=(A.body_iron<0).astype(float).where(A.body_iron.notna())
A['idnoan']=((A.body_iron<0)&(A.LBXHGB>=12)).astype(float).where(A.body_iron.notna()&A.LBXHGB.notna())
A['ida']=((A.body_iron<0)&(A.LBXHGB<12)).astype(float).where(A.body_iron.notna()&A.LBXHGB.notna())
A['fer15']=(A.ferritin<15).astype(float).where(A.ferritin.notna()); A['fer30']=(A.ferritin<30).astype(float).where(A.ferritin.notna())
A['anem']=(A.LBXHGB<12).astype(float).where(A.LBXHGB.notna())
mon={}
for f in ['DEMO_E','DEMO_F','DEMO_G','DEMO_H','DEMO_I','DEMO_J','P_DEMO']:
    x=pd.read_sas(B+'data/raw/'+f+'.xpt',format='xport')[['SEQN','RIDEXMON']]; mon.update(dict(zip(x.SEQN,x.RIDEXMON)))
A['nov_apr']=A.SEQN.map(mon).eq(1).astype(float)
parous=lambda d: ~((d.grp=='ctrl')&(d.RHQ131==2))
SPEC=[s for s in m.OUT if s[5] in ('primary','spline_only','extended')]
PCT={3:[10,50,90],4:[5,35,65,95],5:[5,27.5,50,72.5,95]}

def spline_fit(key,cyc,wt,tr,nk,mask=None,extra=()):
    d,y,w,X=m.prep(key,cyc,wt,tr,extra_mask=mask)
    for e in extra: X[e]=d[e].fillna(0).values
    mth=d.month.where(d.grp=='pp'); knots=np.nanpercentile(mth[w>0],PCT[nk])
    Bm=m.rcs_basis(mth.fillna(0).values,knots); pp=(d.grp=='pp').astype(float).values; X['PP']=pp
    for j in range(Bm.shape[1]): X[f'S{j}']=Bm[:,j]*pp
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    G=m.rcs_basis(GRID,knots); L=np.zeros((len(GRID),len(f.b))); L[:,f.names.index('PP')]=1
    for j in range(Bm.shape[1]): L[:,f.names.index(f'S{j}')]=G[:,j]
    est,se=f.contrast(L); C=L@f.V@L.T
    return f,knots,est,se,C
def metrics(sig):
    sig=np.asarray(sig,bool); init=0
    for i,v in enumerate(sig):
        if v: init=i+1
        else: break
    months=[i+1 for i,v in enumerate(sig) if v]
    later=[x for x in months if x>init]
    return init,(max(months) if months else 0),later
def supt(C,df,B_=200000):
    sd=np.sqrt(np.diag(C)); R=C/np.outer(sd,sd); Lc=np.linalg.cholesky(R+1e-12*np.eye(len(R)))
    Z=rng.standard_normal((B_,len(R)))@Lc.T; Z=Z/np.sqrt(rng.chisquare(df,B_)/df)[:,None]
    return float(np.quantile(np.abs(Z).max(1),.95))
def tr_eff(e,tr): return 100*(np.exp(e)-1) if tr=='log' else e
def fmt(x): return 'none' if not x else ','.join(map(str,x))

# ---------- R1: windowed spline metrics + knot sensitivity (3/4/5 knots) ----------
K=[]
for key,lab,cyc,wt,tr,tier in SPEC:
    main_nk=3 if tier=='spline_only' else 4
    for nk in [3,4,5]:
        f,kn,est,se,C=spline_fit(key,cyc,wt,tr,nk); t=stats.t.ppf(.975,f.dfd)
        sig=(est-t*se>0)|(est+t*se<0); init,last,later=metrics(sig)
        K.append(dict(outcome=lab,var=key,tier=tier,n_knots=nk,main_spec=(nk==main_nk),knots='/'.join(f'{x:.0f}' for x in kn),
                      initial_run_end=init,last_sig_1_22=last,later_isolated_months=fmt(later),pattern=''.join('x' if v else '.' for v in sig)))
K=pd.DataFrame(K); K.to_csv(O+'R1_spline_window_knot_sensitivity.csv',index=False)

# ---------- R2: simultaneous (sup-t) 95% bands, months 1-22, main knots ----------
S=[]; BAND=[]
for key,lab,cyc,wt,tr,tier in SPEC:
    nk=3 if tier=='spline_only' else 4
    f,kn,est,se,C=spline_fit(key,cyc,wt,tr,nk); t=stats.t.ppf(.975,f.dfd); cs=supt(C,f.dfd)
    sp=(est-t*se>0)|(est+t*se<0); ss=(est-cs*se>0)|(est+cs*se<0)
    i1,l1,_=metrics(sp); i2,l2,lat2=metrics(ss)
    S.append(dict(outcome=lab,var=key,tier=tier,design_df=f.dfd,t_pointwise=t,supt_crit=cs,pointwise_initial_run_end=i1,simult_initial_run_end=i2,
                  simult_any_month_excl0=bool(ss.any()),simult_sig_months=fmt([i+1 for i,v in enumerate(ss) if v])))
    for mm,e,s_ in zip(GRID,est,se):
        BAND.append(dict(var=key,month=mm,est=tr_eff(e,tr),lo_point=tr_eff(e-t*s_,tr),hi_point=tr_eff(e+t*s_,tr),lo_simul=tr_eff(e-cs*s_,tr),hi_simul=tr_eff(e+cs*s_,tr)))
pd.DataFrame(S).to_csv(O+'R2_simultaneous_bands_summary.csv',index=False); pd.DataFrame(BAND).to_csv(O+'R2_simultaneous_bands_curves.csv',index=False)

# ---------- R4: S7 parous comparators only ----------
T=m.run_seg_table('S7_parous_comparators',SPEC,extra_mask=parous)
for fam in T.tier.unique():
    i=T.tier==fam; T.loc[i,'q_FDR_family']=m.bh(T.loc[i,'p'])
T.to_csv(O+'R4_S7_parous_comparators_segments.csv',index=False)
BX=[]
for k,lab,var,fnc,cyc,wt in m.BIN+[('bi_neg','Body iron <0 mg/kg','body_iron',lambda v:(v<0),IRONC,'MEC')]:
    rows,nc,pj,pt,dfd,n=m.seg_model(var,cyc,wt,None,family='binomial',ybin=fnc,extra_mask=parous)
    for r in rows: BX.append(dict(outcome=lab,n_ctrl=nc,design_df=dfd,**r,p_joint=pj))
BX=pd.DataFrame(BX); BX['q_FDR']=m.bh(BX.p); BX.to_csv(O+'R4_S7_parous_comparators_binary.csv',index=False)
f,kn,est,se,C=spline_fit('body_iron',IRONC,'MEC',None,4,mask=parous); t=stats.t.ppf(.975,f.dfd)
i_,l_,lat=metrics((est-t*se>0)|(est+t*se<0))
pd.DataFrame([dict(outcome='Body iron spline, parous comparators',initial_run_end=i_,last_sig_1_22=l_,later=fmt(lat))]).to_csv(O+'R4_S7_body_iron_spline.csv',index=False)

# ---------- R6: NCHS proportion standards (Parker et al. 2017) + power ----------
def domain_df(d,ww):
    x=d.loc[ww>0,['STR','PSU']].drop_duplicates(); return x.PSU.nunique()-x.STR.nunique()
def nchs(p,se,n,df):
    if n==0: return dict(flag='suppress',reason='n=0')
    neff=n if se==0 else min(n,p*(1-p)/se**2)
    if df>0 and n>1: neff=min(n,neff*(stats.t.ppf(.975,n-1)/stats.t.ppf(.975,df))**2)
    x=p*neff
    lo=stats.beta.ppf(.025,x,neff-x+1) if x>0 else 0.0; hi=stats.beta.ppf(.975,x+1,neff-x) if x<neff else 1.0
    w=hi-lo; reasons=[]
    if neff<30: reasons.append('n_eff<30')
    if w>=0.30: reasons.append('CI width>=30pp')
    if w>0.05 and p>0 and w/p>1.30: reasons.append('relative CI width>130%')
    flag='suppress' if reasons else ('review (df<8)' if df<8 else 'ok')
    return dict(n_eff=neff,kg_lo=100*lo,kg_hi=100*hi,ci_width_pp=100*w,flag=flag,reason=';'.join(reasons))
R=[]
def prev_cells(var,cyc,wt,groups,family):
    d,y,w,X=m.prep(var,cyc,wt,None); y=d[var].astype(float); w=w*y.notna().astype(float)
    for g,msk in groups(d):
        ww=(w*msk.astype(float)).values; mu,se,n=svymean(y.values,ww,d.STR.values,d.PSU.values)
        ev=int(((y==1)&(ww>0)).sum()); df=domain_df(d,pd.Series(ww,index=d.index))
        R.append(dict(family=family,outcome=var,cell=g,n=n,events=ev,prev_pct=100*mu,se_pct=100*se,rse=se/mu if mu>0 else np.nan,domain_df=df,
                      events_lt10=ev<10,**nchs(mu,se,n,df)))
seggrp=lambda d:[('comparator',d.grp=='ctrl')]+[(s,d.seg==s) for s in SEGS]
bfgrp=lambda d:[('comparator',d.grp=='ctrl')]+[(f'{"BF" if b else "noBF"} {s}',(d.seg==s)&(d.bf==b)) for s in ['0-6','7-12'] for b in (1,0)]
for v in ['bi_neg','idnoan','ida','fer15','fer30','anem']: prev_cells(v,IRONC,'MEC',seggrp,'Table2B_iron_cycles')
for k,lab,var,fnc,cyc,wt in m.BIN:
    A[k+'_b']=fnc(A[var]).astype(float).where(A[var].notna()); prev_cells(k+'_b',cyc,wt,seggrp,'Fig3_TableS3_all_cycles')
prev_cells('bi_neg',IRONC,'MEC',bfgrp,'Breastfeeding_0to12')
R=pd.DataFrame(R); R['dagger']=(R.flag=='suppress')|R.events_lt10; R.to_csv(O+'R6_nchs_reliability_flags.csv',index=False)
# power / precision for body iron <0 (adjusted OR) and absolute difference
A2=pd.read_csv(B+'output/deep/A2_iron_deficiency_prevalence.csv'); o=A2[(A2.outcome.str.startswith('Iron deficiency: body'))&(A2.analysis=='adjusted OR vs control')]
P=[]
for _,r in o.iterrows():
    se=(np.log(r.hi)-np.log(r.lo))/(2*stats.t.ppf(.975,72)); mde=np.exp((stats.t.ppf(.975,72)+stats.norm.ppf(.8))*se)
    P.append(dict(segment=r.group,aOR=r.est,lo=r.lo,hi=r.hi,se_logOR=se,minimal_detectable_OR_80pct=mde))
pd.DataFrame(P).to_csv(O+'R6_power_body_iron_lt0.csv',index=False)

# ---------- R5: 25(OH)D + examination period (Nov-Apr vs May-Oct) ----------
V=['E','F','G','H','I','J']; Q=[]
d,y,w,X=m.prep('LBXVIDMS',V,'MEC',None); X['nov_apr']=d.nov_apr.values
for s in SEGS: X[f'seg={s}']=(d.seg==s).astype(float)
f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns)); t=stats.t.ppf(.975,f.dfd)
for s in SEGS:
    b,se=f.coef(f'seg={s}'); Q.append(dict(model='25(OH)D nmol/L + season',segment=s,est=b,lo=b-t*se,hi=b+t*se,p=2*(1-stats.t.cdf(abs(b/se),f.dfd))))
Q.append(dict(model='joint 3-seg Wald p',p=f.wald([f'seg={s}' for s in SEGS])[2]))
b,se=f.coef('nov_apr'); Q.append(dict(model='season coefficient (Nov-Apr vs May-Oct)',est=b,lo=b-t*se,hi=b+t*se))
d,y,w,X=m.prep('LBXVIDMS',V,'MEC',None); y=(d.LBXVIDMS<50).astype(float).where(d.LBXVIDMS.notna()); X['nov_apr']=d.nov_apr.values
for s in SEGS: X[f'seg={s}']=(d.seg==s).astype(float)
f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'binomial',list(X.columns))
for s in SEGS:
    b,se=f.coef(f'seg={s}'); Q.append(dict(model='25(OH)D<50 aOR + season',segment=s,est=np.exp(b),lo=np.exp(b-t*se),hi=np.exp(b+t*se),p=2*(1-stats.t.cdf(abs(b/se),f.dfd))))
f,kn,est,se,C=spline_fit('LBXVIDMS',V,'MEC',None,3,extra=('nov_apr',)); t=stats.t.ppf(.975,f.dfd)
i_,l_,lat=metrics((est-t*se>0)|(est+t*se<0)); Q.append(dict(model='spline + season',segment=f'initial_run_end={i_}; last_sig_1_22={l_}; sig months={fmt([i+1 for i,v in enumerate((est-t*se>0)|(est+t*se<0)) if v])}'))
pd.DataFrame(Q).to_csv(O+'R5_vitD_season.csv',index=False)

# ---------- R8: RBC folate in nmol/L (code divided nmol/L by 2.265 to get ng/mL) ----------
Fm=pd.read_csv(B+'output/deep/B_folate_mechanism.csv'); Fm=Fm[Fm.analysis.str.contains('ng/mL')].copy()
Fm['est_nmolL']=Fm.est*2.265; Fm['se_nmolL']=Fm.se*2.265; Fm.to_csv(O+'R8_rbc_folate_means_nmolL.csv',index=False)
print('done')
