"""Step1 analytic dataset. Keeps ALL DEMO rows of each cycle (design preserved; domain via weights=0).
Cycles E F G H I P (+ J only for 25(OH)D module). Output: data/step1_analytic.parquet (or pickle)."""
import pandas as pd, numpy as np, os
R='./data/raw/'; D='./data/'
rd=lambda f: pd.read_sas(R+f+'.xpt',format='xport')
fn=lambda m,c: f'P_{m}' if c=='P' else f'{m}_{c}'
ex=lambda f: os.path.exists(R+f+'.xpt')
CYC=['E','F','G','H','I','P','J']
def get(f,cols):
    if not ex(f): return None
    x=rd(f); return x[[c for c in ['SEQN']+cols if c in x.columns]]
frames=[]
for c in CYC:
    d=rd(fn('DEMO',c)) if c!='J' else rd('DEMO_J')
    P=(c=='P')
    d=d.rename(columns={'WTINTPRP':'WTINT','WTMECPRP':'WTMEC','WTINT2YR':'WTINT','WTMEC2YR':'WTMEC'})
    keep=['SEQN','RIAGENDR','RIDAGEYR','RIDEXPRG','RIDRETH1','DMDEDUC2','DMDMARTL','DMDMARTZ','INDFMPIR','SDMVSTRA','SDMVPSU','WTINT','WTMEC']
    m=d[[k for k in keep if k in d.columns]].copy(); m['cycle']=c
    box={'m':m}
    def add(f,cols,ren=None):
        x=get(f,cols)
        if x is None: return
        if ren: x=x.rename(columns=ren)
        x=x[[k for k in x.columns if k=='SEQN' or k not in box['m'].columns]]
        box['m']=box['m'].merge(x,on='SEQN',how='left')
    add(fn('RHQ',c),['RHQ131','RHD143','RHQ162','RHQ171','RHD190','RHQ197','RHQ200'])
    add(fn('SMQ',c),['SMQ020','SMQ040'])
    add(fn('DIQ',c),['DIQ010'])
    add(fn('CBC',c),['LBXHGB','LBXHCT','LBXMCVSI','LBXRDW','LBXWBCSI','LBXPLTSI'])
    add(fn('FERTIN',c),['LBXFER'])
    add(fn('TFR',c),['LBXTFR'])
    add(fn('BIOPRO',c),['LBXSIR','LBXSATSI','LBXSASSI','LBXSGTSI','LBXSCR','LBXSAL','LBXSUA'])
    if P: add('P_FETIB',['LBXIRN'])
    add(fn('FOLATE',c),['LBXRBFSI','LBDRFOSI','WTFOLPRP'])
    add(fn('VID',c),['LBXVIDMS'])
    add(fn('UIO',c),['URXUIO','WTSA2YR','WTSAPRP'],{'WTSAPRP':'WTSA2YR'} if P else None)
    add(fn('TCHOL',c),['LBXTC']); add(fn('HDL',c),['LBDHDD']); add(fn('GHB',c),['LBXGH'])
    add(fn('TRIGLY',c),['WTSAF2YR','WTSAFPRP','LBXTR','LBDLDL'],{'WTSAFPRP':'WTSAF2YR'} if P else None)
    add(fn('GLU',c),['LBXGLU','LBXIN'])
    if c in ('H','I','P'): add(fn('INS',c),['LBXIN'])
    add(fn('ALB_CR',c),['URXUMA','URXUCR'])
    add(fn('CRP',c),['LBXCRP']); add(fn('HSCRP',c),['LBXHSCRP'])
    if P: add('P_BPXO',['BPXOSY1','BPXOSY2','BPXOSY3','BPXODI1','BPXODI2','BPXODI3'])
    else: add(fn('BPX',c),[f'BPXSY{i}' for i in range(1,5)]+[f'BPXDI{i}' for i in range(1,5)])
    add(fn('BMX',c),['BMXBMI','BMXWAIST'])
    add(fn('SLQ',c),['SLD010H','SLD012'])
    add(fn('DPQ',c),[f'DPQ0{i}0' for i in range(1,10)])
    add(fn('PBCD',c),['WTSH2YR','LBXBPB','LBXBCD','LBXTHG'])
    add(fn('DSQTOT',c),['DSD010','DSQTIRON','DSQTFDFE','DSQTFA'])
    add(fn('DR1TOT',c),['WTDRD1','WTDRD1PP','DR1DRSTZ','DR1TIRON','DR1TFDFE','DR1TKCAL'],{'WTDRD1PP':'WTDRD1'} if P else None)
    frames.append(box['m'])
A=pd.concat(frames,ignore_index=True)
# design ids unique across cycles
A['STR']=A.cycle+'_'+A.SDMVSTRA.astype(int).astype(str); A['PSU']=A.STR+'_'+A.SDMVPSU.astype(int).astype(str)
# ---- groups (Step0 definitions) ----
fem=(A.RIAGENDR==2)&A.RIDAGEYR.between(20,44)
A['preg']=(A.RHD143==1)|(A.RIDEXPRG==1)
A['pp']=fem&A.RHQ197.between(1,24)&~A.preg
A['ctrl']=fem&~A.preg&A.RHQ197.isna()&((A.RHQ131==2)|(A.RHQ171==0)|((A.RHD190<=A.RIDAGEYR-3)&(A.RHD190<100)))
A['grp']=np.select([A.pp,A.ctrl],['pp','ctrl'],'other')
A['month']=np.where(A.pp,A.RHQ197,np.nan)
A['seg']=pd.cut(A.month,[0,6,12,24],labels=['0-6','7-12','13-24']).astype(str).where(A.pp,'ctrl')
A['bf']=np.where(A.pp,A.RHQ200.map({1:1,2:0}),np.nan)
A['gdm']=np.where(A.pp,A.RHQ162.map({1:1,2:0}),np.nan)  # borderline(3)/DK -> missing
# ---- covariates ----
A['educ']=A.DMDEDUC2.map({1:'<HS',2:'<HS',3:'HS',4:'>HS',5:'>HS'}).fillna('miss')
A['race']=A.RIDRETH1.map({1:'MexAm',2:'OthHisp',3:'NHW',4:'NHB',5:'Other'})
A['pircat']=pd.cut(A.INDFMPIR,[-1,1.3,3.5,99],labels=['<1.3','1.3-3.5','>=3.5']).astype(str).replace('nan','miss')
mar=np.where(A.cycle=='P',A.DMDMARTZ.isin([1]),A.DMDMARTL.isin([1,6]))
A['married']=mar.astype(int)
par=A.RHQ171.where(A.RHQ171<77); par=par.where(~(A.RHQ131==2),0)
A['parity']=pd.cut(par,[-1,0,1,2,99],labels=['0','1','2','3+']).astype(str).replace('nan','miss')
A['smk']=np.select([A.SMQ020==2,(A.SMQ020==1)&A.SMQ040.isin([1,2]),(A.SMQ020==1)&(A.SMQ040==3)],['never','current','former'],'miss')
# ---- outcomes / harmonisation ----
fer=A.LBXFER.copy()
e=A.cycle=='E'; fer[e]=0.2243+1.079*10**(0.989*np.log10(fer[e])+0.049)   # Hitachi912->E170 (FERTIN_F note) -> e601 (FERTIN_I note)
f=A.cycle=='F'; fer[f]=0.2243+1.079*fer[f]                               # E170 -> e601 (FERTIN_I forward eq.)
A['ferritin']=fer  # I already adjusted by NCHS; P measured on e601
A['stfr']=A.LBXTFR
A['body_iron']=-(np.log10((1.5*A.stfr+0.35)*1000/A.ferritin)-2.8229)/0.1207  # Cook; Roche->Flowers sTfR (Pfeiffer 2007)
A['serum_iron']=A.LBXSIR.where(A.cycle!='P',A.LBXIRN)
A['rbc_folate']=A.LBXRBFSI.fillna(A.LBDRFOSI)
A['crp']=np.where(A.cycle.isin(['E','F']),A.LBXCRP*10,A.LBXHSCRP)  # mg/L
A['uacr']=A.URXUMA/A.URXUCR*100
scr=A.LBXSCR; k=scr/0.7
A['egfr']=142*np.minimum(k,1)**-0.241*np.maximum(k,1)**-1.2*0.9938**A.RIDAGEYR*1.012
A['homa']=A.LBXGLU*A.LBXIN/405
sy=[c for c in A if c.startswith('BPXSY') or c.startswith('BPXOSY')]; di=[c for c in A if c.startswith('BPXDI') or c.startswith('BPXODI')]
A['sbp']=A[sy].mean(1); A['dbp']=A[di].where(A[di]>0).mean(1)
A['sleep']=A.SLD010H.where(A.SLD010H<77).fillna(A.SLD012)
dp=[f'DPQ0{i}0' for i in range(1,10)]; q=A[dp].where(A[dp]<=3)
A['phq9']=q.sum(1,min_count=9)
A['supp_iron']=(A.DSQTIRON>0).astype(float).where(A.DSD010.isin([1,2]))
A['supp_fol']=((A.DSQTFDFE>0)|(A.DSQTFA>0)).astype(float).where(A.DSD010.isin([1,2]))
A['diet_ok']=A.DR1DRSTZ==1
A.to_pickle(D+'step1_analytic.pkl')
S=A[A.grp!='other']
print(S.groupby(['cycle','grp']).size().unstack())
print('pp',A.pp.sum(),'ctrl',A.ctrl.sum())
print(S.groupby('grp')[['educ','pircat','parity','smk']].agg(lambda s:s.value_counts().to_dict()).T.to_string())
print(S[S.pp][['ferritin','stfr','body_iron','rbc_folate','crp','uacr','egfr','sbp','dbp','sleep','phq9','homa']].describe().T)
