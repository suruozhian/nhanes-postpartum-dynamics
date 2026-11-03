"""Step0 feasibility counts: NHANES postpartum 0-24 months (RHQ197).
Cycles with RHQ197: E(2007-08) F(2009-10) G(2011-12) H(2013-14) I(2015-16) P(2017-Mar2020, includes 2017-18 with re-assigned SEQN; J not used to avoid double count).
Output: output/step0_*.csv. No modelling."""
import pandas as pd, numpy as np, os
R='./data/raw/'; O='./output/'
rd=lambda f: pd.read_sas(R+f+'.xpt',format='xport')
CYC=['E','F','G','H','I','P']; YRS={'E':2,'F':2,'G':2,'H':2,'I':2,'P':3.2}; TOT=sum(YRS.values())  # 13.2
fn=lambda m,c: f'P_{m}' if c=='P' else f'{m}_{c}'
rows=[]
for c in CYC:
    d=rd(fn('DEMO',c)); r=rd(fn('RHQ',c))
    wi,wm=('WTINTPRP','WTMECPRP') if c=='P' else ('WTINT2YR','WTMEC2YR')
    d=d[['SEQN','RIAGENDR','RIDAGEYR','RIDEXPRG','RIDEXMON','SDMVSTRA','SDMVPSU',wi,wm]].rename(columns={wi:'WTINT',wm:'WTMEC'})
    keep=['SEQN','RHQ131','RHD143','RHQ162','RHQ171','RHD190','RHQ197','RHQ200']+[v for v in ['RHQ205'] if v in r]
    m=d.merge(r[keep],on='SEQN',how='left'); m['cycle']=c
    m['WTMEC_c']=m.WTMEC*YRS[c]/TOT; m['WTINT_c']=m.WTINT*YRS[c]/TOT
    rows.append(m)
A=pd.concat(rows,ignore_index=True)
A=A[(A.RIAGENDR==2)&(A.RIDAGEYR.between(20,44))]
A['preg']=(A.RHD143==1)|(A.RIDEXPRG==1)
pp=A[A.RHQ197.between(1,24)].copy()
flow={'RHQ197 1-24 valid, F 20-44':len(pp),'RHQ197>24 (dropped)':int((A.RHQ197.between(25,60)).sum()),
      'currently pregnant (RHD143=1 or RIDEXPRG=1, excluded)':int(pp.preg.sum()),
      'RHD143=9 dont know (kept, flag)':int((pp.RHD143==9).sum())}
pp=pp[~pp.preg].copy()
flow['analytic postpartum n']=len(pp); flow['  with MEC exam (WTMEC>0)']=int((pp.WTMEC>0).sum())
pp['seg']=pd.cut(pp.RHQ197,[0,6,12,24],labels=['0-6','7-12','13-24'])
pp['bin3']=pd.cut(pp.RHQ197,np.arange(0,25,3),labels=[f'{a+1}-{a+3}' for a in range(0,24,3)])
pp['bf']=pp.RHQ200.map({1:'BF now',2:'not BF'}).fillna('BF missing')
pp['gdm']=pp.RHQ162.map({1:'GDM yes',2:'GDM no',3:'borderline'}).fillna('GDM missing/DK')
# comparator: non-pregnant 20-44, no RHQ197, nulliparous or last birth >=3 y before current age
cmp_=A[(~A.preg)&(A.RHQ197.isna())&((A.RHQ131==2)|(A.RHQ171==0)|((A.RHD190<=A.RIDAGEYR-3)&(A.RHD190<100)))]
flow['comparator (no birth in ~2y) n']=len(cmp_); flow['  comparator with MEC']=int((cmp_.WTMEC>0).sum())
pd.Series(flow).to_csv(O+'step0_flow.csv',header=['n'])
def tab(df,by):
    g=df.groupby(by,observed=True)
    return pd.DataFrame({'n_unw':g.size(),'n_MEC':g.WTMEC.apply(lambda x:(x>0).sum()),'N_wt_MEC':g.WTMEC_c.sum().round(0)})
T={}
T['cycle']=tab(pp,'cycle'); T['seg']=tab(pp,'seg'); T['bin3']=tab(pp,'bin3'); T['month']=tab(pp,'RHQ197')
T['seg_x_bf']=tab(pp,['seg','bf']); T['seg_x_gdm']=tab(pp,['seg','gdm']); T['cycle_x_seg']=tab(pp,['cycle','seg'])
T['bf_x_gdm']=tab(pp,['gdm','bf'])
for k,v in T.items(): v.to_csv(O+f'step0_n_{k}.csv')
# ---- lab modules: (label, {cycle:(file,var,weight)}) weight None => MEC
MEC=None
L={
 'Hb (CBC)':{c:(fn('CBC',c),'LBXHGB',MEC) for c in CYC},
 'Ferritin':{c:(fn('FERTIN',c),'LBXFER',MEC) for c in ['E','F','I','P']},
 'sTfR':{c:(fn('TFR',c),'LBXTFR',MEC) for c in ['E','F','I','P']},
 'Serum iron':{**{c:(fn('BIOPRO',c),'LBXSIR',MEC) for c in 'EFGHI'},'P':('P_FETIB','LBXIRN',MEC)},
 'RBC folate':{**{c:(fn('FOLATE',c),'LBDRBF',MEC) for c in 'EF'},**{c:(fn('FOLATE',c),'LBDRFO',MEC) for c in 'GHI'},'P':('P_FOLATE','LBDRFO','WTFOLPRP')},
 'Vitamin B12':{c:(fn('VITB12',c),v,MEC) for c,v in [('G','LBXB12'),('H','LBDB12')]},
 '25(OH)D':{c:(fn('VID',c),'LBXVIDMS',MEC) for c in 'EFGHI'},
 'Urinary iodine':{'E':('UIO_E','URXUIO',MEC),**{c:(fn('UIO',c),'URXUIO','WTSA2YR') for c in 'FGHI'},'P':('P_UIO','URXUIO','WTSAPRP')},
 'TSH/FT4':{'E':('THYROD_E','LBXTSH1',MEC),'F':('THYROD_F','LBXTSH1','WTSA2YR'),'G':('THYROD_G','LBXTSH1','WTSA2YR')},
 'TC':{c:(fn('TCHOL',c),'LBXTC',MEC) for c in CYC},
 'HDL-C':{c:(fn('HDL',c),'LBDHDD',MEC) for c in CYC},
 'TG/LDL (fasting)':{c:(fn('TRIGLY',c),'LBXTR','WTSAFPRP' if c=='P' else 'WTSAF2YR') for c in CYC},
 'FPG (fasting)':{c:(fn('GLU',c),'LBXGLU','WTSAFPRP' if c=='P' else 'WTSAF2YR') for c in CYC},
 'Insulin (fasting)':{**{c:('GLU_'+c,'LBXIN','WTSAF2YR') for c in 'EFG'},**{c:('INS_'+c,'LBXIN','WTSAF2YR') for c in 'HI'},'P':('P_INS','LBXIN','WTSAFPRP')},
 'HbA1c':{c:(fn('GHB',c),'LBXGH',MEC) for c in CYC},
 'ALT/AST/Cr (BIOPRO)':{c:(fn('BIOPRO',c),'LBXSATSI',MEC) for c in CYC},
 'CRP/hsCRP':{'E':('CRP_E','LBXCRP',MEC),'F':('CRP_F','LBXCRP',MEC),'I':('HSCRP_I','LBXHSCRP',MEC),'P':('P_HSCRP','LBXHSCRP',MEC)},
 'UACR':{c:(fn('ALB_CR',c),'URXUMA',MEC) for c in CYC},
 'SBP':{**{c:(fn('BPX',c),'BPXSY1',MEC) for c in 'EFGHI'},'P':('P_BPXO','BPXOSY1',MEC)},
 'BMI':{c:(fn('BMX',c),'BMXBMI',MEC) for c in CYC},
}
out=[]; segout=[]
for lab,spec in L.items():
    parts=[]
    for c,(f,v,w) in spec.items():
        x=rd(f); cols=['SEQN',v]+([w] if w else [])
        s=pp[pp.cycle==c].merge(x[cols],on='SEQN',how='inner')
        s=s[s[v].notna()]
        Ty=sum(YRS[k] for k in spec)  # module-specific combined years
        if w: s=s[s[w]>1e-6]; s['W']=s[w]*YRS[c]/Ty
        else: s=s[s.WTMEC>0]; s['W']=s.WTMEC*YRS[c]/Ty
        s['wvar']=w or ('WTMECPRP' if c=='P' else 'WTMEC2YR'); parts.append(s)
    S=pd.concat(parts)
    out.append({'module':lab,'cycles':''.join(spec),'weight':'/'.join(sorted(set(S.wvar))),'n_total':len(S),
                **{f'n_{k}':int((S.seg==k).sum()) for k in ['0-6','7-12','13-24']},
                'min_n_3mo_bin':int(S.bin3.value_counts().min()),'n_BFnow':int((S.bf=='BF now').sum()),'n_GDMyes':int((S.gdm=='GDM yes').sum()),'N_wt':round(S.W.sum())})
LT=pd.DataFrame(out); LT.to_csv(O+'step0_lab_coverage.csv',index=False)
print(pd.Series(flow)); [print('\n##',k,'\n',v) for k,v in T.items() if k!='month']; print(T['month'].T); print(LT.to_string())

# extra diagnostics
pp['agediff']=pp.RIDAGEYR-pp.RHD190
pd.crosstab(pp.cycle,pp.agediff).to_csv(O+'step0_agediff_by_cycle.csv')
print(pd.crosstab(pp.cycle,pp.agediff)); print(pd.crosstab(pp.cycle,pp.seg))
print('month n:',pp.RHQ197.value_counts().sort_index().astype(int).to_dict())
print('RHQ205 (BF>=1mo any child, E-G):',pp[pp.cycle.isin(list('EFG'))].RHQ205.value_counts(dropna=False).to_dict())
print('RHD143==9 by seg',pp[pp.RHD143==9].seg.value_counts().to_dict())
# supplementary: 2017-2018 (J) for 25(OH)D since P has no VID file
rj=rd('RHQ_J'); dj=rd('DEMO_J'); vj=rd('VID_J')
j=dj[['SEQN','RIAGENDR','RIDAGEYR','RIDEXPRG']].merge(rj[['SEQN','RHQ197','RHD143']],on='SEQN').merge(vj[['SEQN','LBXVIDMS']],on='SEQN')
j=j[(j.RIAGENDR==2)&j.RIDAGEYR.between(20,44)&j.RHQ197.between(1,24)&(j.RHD143!=1)&(j.RIDEXPRG!=1)&j.LBXVIDMS.notna()]
print('J 25(OH)D postpartum n:',len(j),pd.cut(j.RHQ197,[0,6,12,24]).value_counts().sort_index().to_dict())
