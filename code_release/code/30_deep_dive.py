"""Deep-dive tables (svy.py, verified identical to R survey 4.5 in docs/r_crosscheck.md).
a) iron-store package (body iron primary), b) RBC-folate mechanism, c) waist x breastfeeding (0-12 mo, descriptive).
Outputs output/deep/*.csv"""
import sys,importlib; sys.path.insert(0,'./code')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
M=importlib.import_module('11_step1_models'); from svy import svyglm, svymean
from scipy import stats
O='./output/deep/'; import os; os.makedirs(O,exist_ok=True)
SEGS=['0-6','7-12','13-24']; IRONC=['E','F','I','P']; MAIN=M.MAIN
A=M.A
A['bi_neg']=(A.body_iron<0).astype(float).where(A.body_iron.notna())
A['idnoan']=((A.body_iron<0)&(A.LBXHGB>=12)).astype(float).where(A.body_iron.notna()&A.LBXHGB.notna())
A['ida']=((A.body_iron<0)&(A.LBXHGB<12)).astype(float).where(A.body_iron.notna()&A.LBXHGB.notna())
A['fer15']=(A.ferritin<15).astype(float).where(A.ferritin.notna())
A['fer30']=(A.ferritin<30).astype(float).where(A.ferritin.notna())
A['anem']=(A.LBXHGB<12).astype(float).where(A.LBXHGB.notna())
A['ldiet_fe']=np.log(A.DR1TIRON.where((A.DR1TIRON>0)&(A.DR1DRSTZ==1)))
A['ldiet_dfe']=np.log(A.DR1TFDFE.where((A.DR1TFDFE>0)&(A.DR1DRSTZ==1)))
A['tot_dfe']=(A.DR1TFDFE.where(A.DR1DRSTZ==1)+A.DSQTFDFE.fillna(0).where(A.DSD010.isin([1,2])))
A['ltot_dfe']=np.log(A.tot_dfe.where(A.tot_dfe>0))
A['rbcf_ngml']=A.rbc_folate/2.265
M.A=A
tc=M.tcrit
def segfit(key,cyc,wt,tr,mask=None,extra=(),family='gaussian',ybin=None,cells=None):
    d,y,w,X=M.prep(key,cyc,wt,tr if ybin is None else None,extra_mask=mask)
    if ybin is not None: y=d[ybin].astype(float)
    for e in extra: X[e]=d[e].values; w=w*d[e].notna().astype(float)
    w=w*np.isfinite(y).astype(float)
    cells=cells or {s:(d.seg==s) for s in SEGS}
    for c,m in cells.items(): X[f'c={c}']=m.astype(float)
    # FIX 2026-10-29 (docs/discrepancy_check.md): restrict the domain to comparator + women in a defined cell for ALL models
    # (previously only binary models). Otherwise, with custom cells (e.g. BF x 0-12 mo), postpartum women outside every cell
    # (13-24 mo) silently joined the reference group. Same domain as Step1 T6. No effect when cells = 3 segments.
    w=w*(d.grp.isin(['ctrl'])|pd.concat(cells,axis=1).any(axis=1)).astype(float)
    f=svyglm(X.fillna(0).values,y.values,w.values,d.STR.values,d.PSU.values,family,list(X.columns))
    out=[]
    for c,m in cells.items():
        b,se=f.coef(f'c={c}'); p=2*(1-stats.t.cdf(abs(b/se),f.dfd)); t=tc(f)
        if family=='binomial': e,lo,hi=np.exp(b),np.exp(b-t*se),np.exp(b+t*se); sc='OR'
        else: e,lo,hi=M.eff(b,se,f,tr); sc='% diff' if tr=='log' else 'abs diff'
        out.append(dict(cell=c,n=int((m&(w>0)).sum()),est=e,lo=lo,hi=hi,p=p,scale=sc))
    n_ctrl=int(((d.grp=='ctrl')&(w>0)).sum())
    for o in out: o['n_ctrl']=n_ctrl; o['df']=f.dfd
    return out,f,d,w
def prev(var,cyc,wt,mask=None,groups=None):
    d,y,w,X=M.prep(var,cyc,wt,None,extra_mask=mask); y=d[var].astype(float); w=w*y.notna().astype(float)
    groups=groups or ([('control',d.grp=='ctrl')]+[(s,d.seg==s) for s in SEGS])
    R=[]
    for g,m in groups:
        ww=(w*m.astype(float)).values; mu,se,n=svymean(y.values,ww,d.STR.values,d.PSU.values)
        R.append(dict(group=g,n=n,n_events=int(((y==1)&(ww>0)).sum()),est=mu*(100 if y.dropna().isin([0,1]).all() else 1),se=se*(100 if y.dropna().isin([0,1]).all() else 1)))
    return R
# ================= a) iron-store package =================
T=[]
def addrows(tag,out,**k):
    for o in out: T.append(dict(analysis=tag,**k,**o))
noFe=lambda d:d.supp_iron==0; crp5=lambda d:d.crp<=5
for key,tr,lab in [('body_iron',None,'Body iron (mg/kg)'),('ferritin','log','Ferritin'),('stfr','log','sTfR')]:
    addrows('A0 main (MEC wt)',segfit(key,IRONC,'MEC',tr)[0],outcome=lab)
    addrows('A1 excl iron-supplement users',segfit(key,IRONC,'MEC',tr,mask=noFe)[0],outcome=lab)
    addrows('A2 excl CRP>5 mg/L',segfit(key,IRONC,'MEC',tr,mask=crp5)[0],outcome=lab)
    addrows('A3 excl iron-suppl AND CRP>5',segfit(key,IRONC,'MEC',tr,mask=lambda d:(d.supp_iron==0)&(d.crp<=5))[0],outcome=lab)
    addrows('A4a diet-iron sample, unadjusted (DR1 wt)',segfit(key,IRONC,'DR1',tr,mask=lambda d:d.ldiet_fe.notna())[0],outcome=lab)
    addrows('A4b + log dietary iron (DR1 wt)',segfit(key,IRONC,'DR1',tr,extra=('ldiet_fe',))[0],outcome=lab)
    addrows('A5 + iron-supplement use as covariate',segfit(key,IRONC,'MEC',tr,extra=('supp_iron',))[0],outcome=lab)
pd.DataFrame(T).to_csv(O+'A1_iron_store_segment_contrasts.csv',index=False)
# prevalence
P=[]
for var,lab in [('bi_neg','Iron deficiency: body iron <0 mg/kg'),('idnoan','Iron deficiency WITHOUT anemia (body iron<0 & Hb>=12)'),('ida','Iron-deficiency anemia (body iron<0 & Hb<12)'),
                ('fer15','Ferritin <15 ng/mL'),('fer30','Ferritin <30 ng/mL'),('anem','Anemia Hb<12 g/dL (iron cycles)')]:
    for r in prev(var,IRONC,'MEC'): P.append(dict(outcome=lab,analysis='all',**r))
    for r in prev(var,IRONC,'MEC',mask=crp5): P.append(dict(outcome=lab,analysis='excl CRP>5',**r))
    o,_,_,_=segfit(var,IRONC,'MEC',None,family='binomial',ybin=var)
    for x in o: P.append(dict(outcome=lab,analysis='adjusted OR vs control',group=x['cell'],n=x['n'],est=x['est'],lo=x['lo'],hi=x['hi'],p=x['p']))
pd.DataFrame(P).to_csv(O+'A2_iron_deficiency_prevalence.csv',index=False)
# breastfeeding description (0-12)
BF=[]
for key,tr,lab in [('body_iron',None,'Body iron (mg/kg)'),('ferritin','log','Ferritin'),('stfr','log','sTfR')]:
    d0,_,_,_=M.prep(key,IRONC,'MEC',tr)
    cells={f'{"BF" if b else "noBF"} {s}':((d0.seg==s)&(d0.bf==b)) for s in ['0-6','7-12'] for b in (1,0)}
    o,f,d,w=segfit(key,IRONC,'MEC',tr,cells=cells)
    for x in o: BF.append(dict(outcome=lab,contrast='vs control (adjusted)',**x))
    for s in ['0-6','7-12']:
        L=np.zeros(len(f.b)); L[f.names.index(f'c=BF {s}')]=1; L[f.names.index(f'c=noBF {s}')]=-1
        e,se=f.contrast(L[None,:]); e,se=e[0],se[0]; ee,lo,hi=M.eff(e,se,f,tr)
        BF.append(dict(outcome=lab,contrast='BF vs noBF (adjusted)',cell=s,est=ee,lo=lo,hi=hi,p=2*(1-stats.t.cdf(abs(e/se),f.dfd))))
for r in prev('bi_neg',IRONC,'MEC',groups=None if False else None): pass
d0,_,_,_=M.prep('bi_neg',IRONC,'MEC',None)
grp=[('control',d0.grp=='ctrl')]+[(f'{"BF" if b else "noBF"} {s}',(d0.seg==s)&(d0.bf==b)) for s in ['0-6','7-12'] for b in (1,0)]
for r in prev('bi_neg',IRONC,'MEC',groups=grp): BF.append(dict(outcome='Body iron <0 prevalence %',contrast='weighted prevalence',cell=r['group'],n=r['n'],est=r['est'],se=r['se'],n_events=r['n_events']))
pd.DataFrame(BF).to_csv(O+'A3_iron_by_breastfeeding_0to12mo.csv',index=False)
# ================= b) folate mechanism =================
F=[]
for sv,lab in [(1,'supplement users (folate/folic acid, 30-d DSQ)'),(0,'non-users')]:
    o,_,_,_=segfit('rbc_folate',MAIN,'FOL','log',mask=lambda d,sv=sv:d.supp_fol==sv)
    for x in o: F.append(dict(analysis='B1 stratified by supplement use; vs control of same stratum',stratum=lab,**x))
    for r in prev('rbcf_ngml',MAIN,'FOL',mask=lambda d,sv=sv:d.supp_fol==sv): F.append(dict(analysis='B1 weighted mean RBC folate ng/mL (unadjusted)',stratum=lab,cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
# interaction supplement x segment
d,y,w,X=M.prep('rbc_folate',MAIN,'FOL','log'); w=w*d.supp_fol.notna().astype(float); X['supp']=d.supp_fol.fillna(0).values
for s in SEGS: X[f'seg={s}']=(d.seg==s).astype(float); X[f'supp_x_{s}']=X[f'seg={s}']*X.supp
f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
pint=f.wald([f'supp_x_{s}' for s in SEGS])[2]
F.append(dict(analysis='B2 supplement x segment interaction (Wald, 3 df)',p=pint,n=int((w>0).sum())))
# supplement prevalence by segment
for r in prev('supp_fol',MAIN,'MEC'): F.append(dict(analysis='B3 prevalence of folate-supplement use % (30-d)',cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
for key,lab in [('DR1TFDFE','dietary DFE'),('tot_dfe','total DFE (diet+supplement)')]:
    for r in prev(key,MAIN,'DR1'): F.append(dict(analysis=f'B3 weighted mean {lab} ug/d',cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
# intake adjustment (same sample comparisons)
F+= [dict(analysis='B4a intake sample, unadjusted (DR1 wt)',**x) for x in segfit('rbc_folate',MAIN,'DR1','log',mask=lambda d:d.ltot_dfe.notna()&d.ldiet_dfe.notna())[0]]
F+= [dict(analysis='B4b + log dietary DFE',**x) for x in segfit('rbc_folate',MAIN,'DR1','log',extra=('ldiet_dfe',),mask=lambda d:d.ltot_dfe.notna())[0]]
F+= [dict(analysis='B4c + log total DFE (diet+supplement)',**x) for x in segfit('rbc_folate',MAIN,'DR1','log',extra=('ltot_dfe',),mask=lambda d:d.ldiet_dfe.notna())[0]]
F+= [dict(analysis='B4d + log total DFE + supplement use',**x) for x in segfit('rbc_folate',MAIN,'DR1','log',extra=('ltot_dfe','supp_fol'),mask=lambda d:d.ldiet_dfe.notna())[0]]
F+= [dict(analysis='B5 cycles I+P only (2015-Mar2020)',**x) for x in segfit('rbc_folate',['I','P'],'FOL','log')[0]]
# Vladutiu 2021 comparison: 2007-2016 (E-I), first year postpartum (1-12 mo) vs comparator, ng/mL
d0,_,_,_=M.prep('rbcf_ngml',['E','F','G','H','I'],'FOL',None)
grp=[('control (nullip or >=3y)',d0.grp=='ctrl'),('1-12 mo (first year)',(d0.month<=12)&(d0.grp=='pp')),('1-6 mo',d0.seg=='0-6'),('7-12 mo',d0.seg=='7-12'),('13-24 mo',d0.seg=='13-24')]
for r in prev('rbcf_ngml',['E','F','G','H','I'],'FOL',groups=grp): F.append(dict(analysis='B6 Vladutiu-comparable: unadjusted weighted mean RBC folate ng/mL, cycles 2007-2016',cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
pd.DataFrame(F).to_csv(O+'B_folate_mechanism.csv',index=False)
# ================= c) waist x breastfeeding =================
C=[]
d0,_,_,_=M.prep('BMXWAIST',MAIN,'MEC',None)
cells={f'{"BF" if b else "noBF"} {s}':((d0.seg==s)&(d0.bf==b)) for s in ['0-6','7-12'] for b in (1,0)}
for tag,extra in [('C1 adjusted (main covariates)',()),('C2 + BMI (waist beyond BMI)',('BMXBMI',))]:
    o,f,d,w=segfit('BMXWAIST',MAIN,'MEC',None,extra=extra,cells=cells)
    for x in o: C.append(dict(analysis=tag,contrast='vs control',**x))
    for s in ['0-6','7-12']:
        L=np.zeros(len(f.b)); L[f.names.index(f'c=BF {s}')]=1; L[f.names.index(f'c=noBF {s}')]=-1
        e,se=f.contrast(L[None,:]); C.append(dict(analysis=tag,contrast='BF vs noBF',cell=s,est=e[0],lo=e[0]-tc(f)*se[0],hi=e[0]+tc(f)*se[0],p=2*(1-stats.t.cdf(abs(e[0]/se[0]),f.dfd))))
for v,lab in [('BMXWAIST','waist cm'),('BMXBMI','BMI')]:
    grp=[('control',d0.grp=='ctrl')]+list(cells.items())
    for r in prev(v,MAIN,'MEC',groups=grp): C.append(dict(analysis=f'C3 weighted mean {lab} (unadjusted)',cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
A['w88']=(A.BMXWAIST>=88).astype(float).where(A.BMXWAIST.notna()); M.A=A
for r in prev('w88',MAIN,'MEC',groups=[('control',d0.grp=='ctrl')]+list(cells.items())): C.append(dict(analysis='C4 prevalence waist>=88 cm %',cell=r['group'],n=r['n'],est=r['est'],se=r['se']))
pd.DataFrame(C).to_csv(O+'C_waist_by_breastfeeding_0to12mo.csv',index=False)
for fn in ['A1_iron_store_segment_contrasts','A2_iron_deficiency_prevalence','A3_iron_by_breastfeeding_0to12mo','B_folate_mechanism','C_waist_by_breastfeeding_0to12mo']:
    print('=====',fn); print(pd.read_csv(O+fn+'.csv').round(3).to_string())
