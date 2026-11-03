"""Step1 models: design-weighted RCS of months since delivery + 3 segments vs comparator.
Outputs CSV in output/step1/. No formal figures (draft diagnostics via 12_step1_diagnostic_plots.py)."""
import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'./code')
from svy import svyglm, svymean, wquantile, rcs_basis
from scipy import stats
B='./'; O=B+'output/step1/'
A=pd.read_pickle(B+'data/step1_analytic.pkl').copy()
YRS={'E':2,'F':2,'G':2,'H':2,'I':2,'P':3.2,'J':2}
MAIN=['E','F','G','H','I','P']
# collapse 1-2-person missing categories (documented)
A.loc[A.smk=='miss','smk']='never'; A.loc[A.parity=='miss','parity']='1'; A.loc[A.educ=='miss','educ']='>HS'
PSU_ALL=A[['cycle','STR','PSU']].drop_duplicates()
A=A[A.grp!='other'].copy()
# dummy zero-weight rows so that every PSU of the design is present (== survey::subset semantics)
dummy=PSU_ALL.copy(); dummy['grp']='dummy'
A=pd.concat([A,dummy],ignore_index=True)

def W(df,wt,cycles):
    T=sum(YRS[c] for c in cycles); w=pd.Series(0.0,index=df.index)
    for c in cycles:
        i=df.cycle==c
        if wt=='MEC': base=df.WTMEC
        elif wt=='INT': base=df.WTINT
        elif wt=='SAF': base=df.WTSAF2YR
        elif wt=='SA': base=df.WTMEC if c=='E' else df.WTSA2YR
        elif wt=='FOL': base=df.WTFOLPRP if c=='P' else df.WTMEC
        elif wt=='SH': base=df.WTSH2YR if c in ('H','I') else df.WTMEC
        elif wt=='DR1': base=df.WTDRD1
        w[i]=base[i].fillna(0)*YRS[c]/T
    return w.where(w>1e-6,0.0)   # subsample/fasting weights: keep only >1e-6

# outcome specs: key, label, cycles, weight, transform, tier
OUT=[
 ('LBXHGB','Hb g/dL',MAIN,'MEC',None,'primary'),('LBXMCVSI','MCV fL',MAIN,'MEC',None,'primary'),
 ('LBXRDW','RDW %',MAIN,'MEC',None,'primary'),('LBXWBCSI','WBC 10^9/L',MAIN,'MEC',None,'primary'),
 ('LBXPLTSI','Platelets 10^9/L',MAIN,'MEC',None,'primary'),
 ('ferritin','Ferritin ng/mL (harmonised e601)',['E','F','I','P'],'MEC','log','primary'),
 ('stfr','sTfR mg/L',['E','F','I','P'],'MEC','log','primary'),
 ('body_iron','Body iron mg/kg',['E','F','I','P'],'MEC',None,'primary'),
 ('serum_iron','Serum iron ug/dL',MAIN,'MEC',None,'primary'),
 ('rbc_folate','RBC folate nmol/L',MAIN,'FOL','log','primary'),
 ('LBXGH','HbA1c %',MAIN,'MEC',None,'primary'),
 ('LBXTC','Total cholesterol mg/dL',MAIN,'MEC',None,'primary'),('LBDHDD','HDL-C mg/dL',MAIN,'MEC',None,'primary'),
 ('LBXSATSI','ALT U/L',MAIN,'MEC','log','primary'),('LBXSASSI','AST U/L',MAIN,'MEC','log','primary'),
 ('LBXSGTSI','GGT U/L',MAIN,'MEC','log','primary'),('LBXSAL','Albumin g/dL',MAIN,'MEC',None,'primary'),
 ('LBXSCR','Creatinine mg/dL',MAIN,'MEC',None,'primary'),('egfr','eGFR CKD-EPI2021',MAIN,'MEC',None,'primary'),
 ('LBXSUA','Uric acid mg/dL',MAIN,'MEC',None,'primary'),
 ('uacr','UACR mg/g',MAIN,'MEC','log','primary'),('crp','CRP/hsCRP mg/L',['E','F','I','P'],'MEC','log','primary'),
 ('sbp','SBP mmHg',MAIN,'MEC',None,'primary'),('dbp','DBP mmHg',MAIN,'MEC',None,'primary'),
 ('BMXBMI','BMI kg/m2',MAIN,'MEC',None,'primary'),
 ('LBXTR','TG mg/dL (fasting)',MAIN,'SAF','log','spline_only'),('LBDLDL','LDL-C mg/dL (fasting)',MAIN,'SAF',None,'spline_only'),
 ('LBXGLU','FPG mg/dL (fasting)',MAIN,'SAF',None,'spline_only'),('LBXIN','Insulin uU/mL (fasting)',MAIN,'SAF','log','spline_only'),
 ('homa','HOMA-IR (fasting)',MAIN,'SAF','log','spline_only'),
 ('URXUIO','Urinary iodine ug/L',MAIN,'SA','log','spline_only'),
 ('LBXVIDMS','25(OH)D nmol/L (E-I + J)',['E','F','G','H','I','J'],'MEC',None,'spline_only'),
 ('sleep','Sleep h/night',MAIN,'INT',None,'extended'),('phq9','PHQ-9 score',MAIN,'MEC',None,'extended'),
 ('BMXWAIST','Waist cm',MAIN,'MEC',None,'extended'),
 ('DR1TIRON','Diet iron mg/d (day1)',MAIN,'DR1','log','extended_diet'),('DR1TFDFE','Diet folate DFE ug/d (day1)',MAIN,'DR1','log','extended_diet'),
 ('DR1TKCAL','Energy kcal/d (day1)',MAIN,'DR1',None,'extended_diet'),
 ('LBXBPB','Blood lead ug/dL',MAIN,'SH','log','appendix_metal'),('LBXBCD','Blood cadmium ug/L',MAIN,'SH','log','appendix_metal'),
 ('LBXTHG','Blood total Hg ug/L',MAIN,'SH','log','appendix_metal'),
]
BIN=[('anemia','Anemia Hb<12','LBXHGB',lambda v:(v<12),MAIN,'MEC'),
     ('iron_def','Iron deficiency ferritin<15','ferritin',lambda v:(v<15),['E','F','I','P'],'MEC'),
     ('rbcf_low','RBC folate <748 nmol/L','rbc_folate',lambda v:(v<748),MAIN,'FOL'),
     ('vitd_low','25(OH)D <50 nmol/L','LBXVIDMS',lambda v:(v<50),['E','F','G','H','I','J'],'MEC'),
     ('a1c_pre','HbA1c >=5.7%','LBXGH',lambda v:(v>=5.7),MAIN,'MEC'),
     ('obese','BMI>=30','BMXBMI',lambda v:(v>=30),MAIN,'MEC'),
     ('waist88','Waist>=88cm','BMXWAIST',lambda v:(v>=88),MAIN,'MEC'),
     ('phq10','PHQ-9>=10','phq9',lambda v:(v>=10),MAIN,'MEC'),
     ('short_sleep','Sleep<7h','sleep',lambda v:(v<7),MAIN,'INT')]

def covmat(d,cycles,pir='cat'):
    X=pd.DataFrame(index=d.index); X['const']=1.0
    a=d.RIDAGEYR-32; X['age']=a; X['age2']=a**2
    for v,ref in [('race','NHW'),('educ','>HS'),('parity','0'),('smk','never')]+([('pircat','1.3-3.5')] if pir=='cat' else []):
        for lev in sorted(d[v].dropna().unique()):
            if lev!=ref: X[f'{v}={lev}']=(d[v]==lev).astype(float)
    if pir=='cont': X['pir']=d.INDFMPIR
    X['married']=d.married.astype(float)
    for c in cycles[1:]: X[f'cyc={c}']=(d.cycle==c).astype(float)
    return X

def prep(key,cycles,wt,tr,extra_mask=None,pir='cat'):
    d=A[A.cycle.isin(cycles)].copy()
    y=d[key].astype(float)
    if tr=='log': y=np.log(y.where(y>0))
    w=W(d,wt,cycles)*(d.grp.isin(['pp','ctrl'])).astype(float)
    if extra_mask is not None: w=w*extra_mask(d).astype(float)
    X=covmat(d.fillna({'RIDAGEYR':32}),cycles,pir)
    if pir=='cont': w=w*d.INDFMPIR.notna().astype(float)
    w=w*np.isfinite(y).astype(float)
    return d,y,w,X

tcrit=lambda f: stats.t.ppf(.975,f.dfd)
def eff(b,se,f,tr):
    lo,hi=b-tcrit(f)*se,b+tcrit(f)*se
    if tr=='log': return 100*(np.exp(b)-1),100*(np.exp(lo)-1),100*(np.exp(hi)-1)
    return b,lo,hi

def seg_model(key,cycles,wt,tr,family='gaussian',ybin=None,**kw):
    d,y,w,X=prep(key,cycles,wt,tr if ybin is None else None,**kw)
    if ybin is not None: y=ybin(d[key]).astype(float).where(d[key].notna())
    for s in ['0-6','7-12','13-24']: X[f'seg={s}']=(d.seg==s).astype(float)
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,family,list(X.columns))
    rows=[]
    for s in ['0-6','7-12','13-24']:
        b,se=f.coef(f'seg={s}'); p=2*(1-stats.t.cdf(abs(b/se),f.dfd))
        e,lo,hi=(eff(b,se,f,tr) if ybin is None else (np.exp(b),np.exp(b-tcrit(f)*se),np.exp(b+tcrit(f)*se)))
        rows.append({'segment':s,'n_seg':int(((d.seg==s)&(w>0)).sum()),'est':e,'lo':lo,'hi':hi,'p':p})
    n_ctrl=int(((d.grp=='ctrl')&(w>0)).sum())
    Fj,q,pj=f.wald([f'seg={s}' for s in ['0-6','7-12','13-24']])
    Ft,qt,pt=f.wald(L=np.array([[0]*(len(f.b)-3)+[1,-1,0],[0]*(len(f.b)-3)+[0,1,-1]]))  # heterogeneity across segments
    return rows,n_ctrl,pj,pt,f.dfd,int((w>0).sum())

def spline_model(key,cycles,wt,tr,nk):
    d,y,w,X=prep(key,cycles,wt,tr)
    m=d.month.where(d.grp=='pp')
    knots=np.nanpercentile(m[w>0],{4:[5,35,65,95],3:[10,50,90]}[nk])
    Bm=rcs_basis(m.fillna(0).values,knots); pp=(d.grp=='pp').astype(float).values
    X['PP']=pp
    for j in range(Bm.shape[1]): X[f'S{j}']=Bm[:,j]*pp
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    grid=np.arange(1,25); G=rcs_basis(grid,knots); k=len(f.b); iPP=f.names.index('PP'); iS=[f.names.index(f'S{j}') for j in range(Bm.shape[1])]
    L=np.zeros((24,k)); L[:,iPP]=1; L[:,iS]=G
    est,se=f.contrast(L); curve=[]
    for mth,e_,s_ in zip(grid,est,se):
        e,lo,hi=eff(e_,s_,f,tr); curve.append({'month':mth,'est':e,'lo':lo,'hi':hi,'sig':(lo>0) or (hi<0) if tr!='log' else (lo>0) or (hi<0)})
    p_overall=f.wald(['PP']+[f'S{j}' for j in range(Bm.shape[1])])[2]
    p_trend=f.wald([f'S{j}' for j in range(Bm.shape[1])])[2]
    p_nonlin=f.wald([f'S{j}' for j in range(1,Bm.shape[1])])[2] if Bm.shape[1]>1 else np.nan
    sig=[c['sig'] for c in curve]; last_sig=max([c['month'] for c in curve if c['sig']],default=0)
    # first month from which CI includes 0 for all later months
    ret=next((mm for mm in range(1,25) if not any(sig[mm-1:])),None)
    return curve,dict(knots='/'.join(f'{x:.0f}' for x in knots),p_overall=p_overall,p_time_trend=p_trend,p_nonlinear=p_nonlin,
                      last_sig_month=last_sig,returns_to_ctrl_from_month=ret,n=int((w>0).sum()))

def bh(p):
    p=np.asarray(p,float); n=np.isfinite(p).sum(); q=np.full_like(p,np.nan); idx=np.where(np.isfinite(p))[0]
    o=idx[np.argsort(p[idx])]; r=np.arange(1,n+1); qq=np.minimum.accumulate((p[o]*n/r)[::-1])[::-1]; q[o]=np.minimum(qq,1); return q

def run_seg_table(tag,specs,**kw):
    R=[]
    for key,lab,cyc,wt,tr,tier in specs:
        cyc2=[c for c in cyc if c in kw.get('only_cycles',cyc)]
        if len(cyc2)==0: continue
        kk={k:v for k,v in kw.items() if k!='only_cycles'}
        rows,nc,pj,pt,dfd,n=seg_model(key,cyc2,wt,tr,**kk)
        for r in rows: R.append({'analysis':tag,'outcome':lab,'var':key,'tier':tier,'scale':'% diff' if tr=='log' else 'abs diff','cycles':''.join(cyc2),'weight':wt,'n_ctrl':nc,'n_total':n,'design_df':dfd,**r,'p_joint_3seg':pj,'p_heterog_seg':pt})
    return pd.DataFrame(R)

if __name__=="__main__":
    # ---------- Table 1: weighted characteristics ----------
    d=A[A.cycle.isin(MAIN)].copy(); d['w']=W(d,'MEC',MAIN)
    t1=[]
    for g,sub in [('ctrl',d[d.grp=='ctrl'])]+[(s,d[d.seg==s]) for s in ['0-6','7-12','13-24']]:
        sub=sub[sub.w>0]; ws=sub.w
        row={'group':g,'n':len(sub),'N_wt':round(ws.sum()),'age_mean':np.average(sub.RIDAGEYR,weights=ws)}
        for v in ['race','educ','pircat','parity','smk']:
            for lev,val in (sub.groupby(v).w.sum()/ws.sum()*100).items(): row[f'{v}={lev} %']=round(val,1)
        row['married %']=round(np.average(sub.married,weights=ws)*100,1)
        if g!='ctrl':
            row['BF now %']=round(np.average(sub.bf.fillna(0),weights=ws)*100,1); row['GDM hx %']=round(np.average(sub.gdm.fillna(0),weights=ws)*100,1)
            row['n BF now']=int((sub.bf==1).sum()); row['n GDM']=int((sub.gdm==1).sum())
        t1.append(row)
    pd.DataFrame(t1).T.to_csv(O+'T1_characteristics_weighted.csv',header=False)

    # ---------- Table 2: segment contrasts (main) ----------
    T2=run_seg_table('main',OUT)
    for fam in ['primary','spline_only','extended','extended_diet','appendix_metal']:
        i=T2.tier==fam; T2.loc[i,'q_FDR_family']=bh(T2.loc[i,'p'])
    T2.to_csv(O+'T2_segment_vs_control_adjusted.csv',index=False)

    # ---------- Table 3: RCS curves ----------
    C=[];S=[]
    for key,lab,cyc,wt,tr,tier in OUT:
        nk=3 if tier in ('spline_only','appendix_metal','extended_diet') else 4
        curve,summ=spline_model(key,cyc,wt,tr,nk)
        for c in curve: C.append({'outcome':lab,'var':key,'tier':tier,'scale':'% diff' if tr=='log' else 'abs diff',**c})
        S.append({'outcome':lab,'var':key,'tier':tier,'n_knots':nk,**summ})
    S=pd.DataFrame(S)
    for fam in S.tier.unique():
        i=S.tier==fam; S.loc[i,'q_overall_FDR']=bh(S.loc[i,'p_overall']); S.loc[i,'q_trend_FDR']=bh(S.loc[i,'p_time_trend'])
    pd.DataFrame(C).to_csv(O+'T3a_rcs_curve_diff_vs_control.csv',index=False); S.to_csv(O+'T3b_rcs_tests.csv',index=False)

    # ---------- Table 4: unadjusted weighted means / medians by segment & 3-mo bins ----------
    T4=[]
    for key,lab,cyc,wt,tr,tier in OUT:
        d=A[A.cycle.isin(cyc)].copy(); w=W(d,wt,cyc); y=d[key].astype(float)
        d['bin3']=pd.cut(d.month,np.arange(0,25,3),labels=[f'{a+1}-{a+3}' for a in range(0,24,3)]).astype(str)
        groups=[('ctrl',d.grp=='ctrl')]+[(s,d.seg==s) for s in ['0-6','7-12','13-24']]
        if tier=='primary': groups+= [(b,d.bin3==b) for b in [f'{a+1}-{a+3}' for a in range(0,24,3)]]
        for g,msk in groups:
            ww=(w*msk.astype(float)*y.notna().astype(float)).values
            mu,se,n=svymean(y.values,ww,d.STR.values,d.PSU.values)
            T4.append({'outcome':lab,'var':key,'tier':tier,'group':g,'n':n,'wmean':mu,'se':se,'wmedian':wquantile(y.values,ww,.5) if n else np.nan,'flag_n<30':n<30})
    pd.DataFrame(T4).to_csv(O+'T4_weighted_means_by_segment_bin.csv',index=False)

    # ---------- Table 5: binary outcomes ----------
    T5=[]
    for k,lab,var,fnc,cyc,wt in BIN:
        rows,nc,pj,pt,dfd,n=seg_model(var,cyc,wt,None,family='binomial',ybin=fnc)
        d=A[A.cycle.isin(cyc)].copy(); w=W(d,wt,cyc); yb=fnc(d[var]).astype(float).where(d[var].notna())
        prev={}
        for g,msk in [('ctrl',d.grp=='ctrl')]+[(s,d.seg==s) for s in ['0-6','7-12','13-24']]:
            ww=(w*msk.astype(float)*yb.notna().astype(float)).values; mu,se,nn=svymean(yb.values,ww,d.STR.values,d.PSU.values)
            prev[g]=(100*mu,100*se,nn)
        for r in rows:
            T5.append({'outcome':lab,'segment':r['segment'],'n_seg':r['n_seg'],'prev_seg_%':prev[r['segment']][0],'se_%':prev[r['segment']][1],
                       'prev_ctrl_%':prev['ctrl'][0],'n_ctrl':prev['ctrl'][2],'adj_OR':r['est'],'lo':r['lo'],'hi':r['hi'],'p':r['p'],'p_joint':pj})
    T5=pd.DataFrame(T5); T5['q_FDR']=bh(T5.p); T5.to_csv(O+'T5_binary_prevalence_OR.csv',index=False)
    print('main done')
