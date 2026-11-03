"""Step1: breastfeeding stratification (0-12 mo only), GDM exploratory interaction (full 1-24), sensitivity analyses."""
import sys; sys.path.insert(0,'./code')
import importlib; M=importlib.import_module('11_step1_models')
from svy import svyglm, rcs_basis
import numpy as np, pandas as pd
from scipy import stats
A,W,OUT,O,MAIN,prep,eff,bh,run_seg_table,tcrit=M.A,M.W,M.OUT,M.O,M.MAIN,M.prep,M.eff,M.bh,M.run_seg_table,M.tcrit
PRIM=[o for o in OUT if o[5]=='primary']; EXT=[o for o in OUT if o[5]=='extended']
# ---------- breastfeeding (0-12 mo) ----------
R=[]
for key,lab,cyc,wt,tr,tier in PRIM+EXT:
    d,y,w,X=prep(key,cyc,wt,tr)
    keep=(d.grp=='ctrl')|((d.grp=='pp')&(d.month<=12)&d.bf.notna()); w=w*keep.astype(float)
    cells=[(b,s) for b in (1,0) for s in ('0-6','7-12')]
    for b,s in cells: X[f'bf{b}_{s}']=((d.seg==s)&(d.bf==b)).astype(float)
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    k=len(f.b); ix={c:f.names.index(f'bf{c[0]}_{c[1]}') for c in cells}
    for s in ('0-6','7-12'):
        for b in (1,0):
            be,se=f.coef(f'bf{b}_{s}'); e,lo,hi=eff(be,se,f,tr)
            R.append({'outcome':lab,'tier':tier,'segment':s,'contrast':f"{'BF' if b else 'noBF'} vs ctrl",'n':int(((d.seg==s)&(d.bf==b)&(w>0)).sum()),'est':e,'lo':lo,'hi':hi,'p':2*(1-stats.t.cdf(abs(be/se),f.dfd))})
        L=np.zeros(k); L[ix[(1,s)]]=1; L[ix[(0,s)]]=-1; be,se=f.contrast(L[None,:]); be,se=be[0],se[0]; e,lo,hi=eff(be,se,f,tr)
        R.append({'outcome':lab,'tier':tier,'segment':s,'contrast':'BF vs noBF','n':np.nan,'est':e,'lo':lo,'hi':hi,'p':2*(1-stats.t.cdf(abs(be/se),f.dfd))})
    L=np.zeros(k); L[ix[(1,'0-6')]]=1; L[ix[(0,'0-6')]]=-1; L[ix[(1,'7-12')]]=-1; L[ix[(0,'7-12')]]=1
    R[-1]['p_interaction_BFxseg']=f.wald(L=L[None,:])[2]
T=pd.DataFrame(R); i=T.contrast=='BF vs noBF'; T.loc[i,'q_FDR']=bh(T.loc[i,'p']); T.to_csv(O+'T6_breastfeeding_0to12mo.csv',index=False)
# ---------- GDM exploratory (pp only, 1-24) ----------
G=[]
for key,lab,cyc,wt,tr,tier in OUT:
    if tier=='appendix_metal' or tier=='extended_diet': continue
    d,y,w,X=prep(key,cyc,wt,tr); w=w*((d.grp=='pp')&d.gdm.notna()).astype(float)
    if (w>0).sum()<60: continue
    X=X.drop(columns=[c for c in X if c.startswith('parity=0')])
    m=d.month.fillna(12); kn=np.nanpercentile(m[w>0],[10,50,90]); Bm=rcs_basis(m.values,kn)
    for j in range(Bm.shape[1]): X[f'S{j}']=Bm[:,j]
    X['gdm']=d.gdm.fillna(0).values; X['gdm_x_m']=X.gdm*(m.values-12)
    f=svyglm(X.values,y.values,w.values,d.STR.values,d.PSU.values,'gaussian',list(X.columns))
    be,se=f.coef('gdm'); e,lo,hi=eff(be,se,f,tr); bi,si=f.coef('gdm_x_m')
    G.append({'outcome':lab,'tier':tier,'n_pp':int((w>0).sum()),'n_gdm':int(((d.gdm==1)&(w>0)).sum()),'gdm_diff_at_12mo':e,'lo':lo,'hi':hi,
              'p_gdm_main':2*(1-stats.t.cdf(abs(be/se),f.dfd)),'gdm_x_month_slope':bi,'p_gdm_x_month':2*(1-stats.t.cdf(abs(bi/si),f.dfd)),
              'p_joint_gdm':f.wald(['gdm','gdm_x_m'])[2]})
G=pd.DataFrame(G); G['q_joint_FDR']=bh(G.p_joint_gdm); G.to_csv(O+'T7_gdm_exploratory_interaction.csv',index=False)
# ---------- sensitivity ----------
IRON=['LBXHGB','LBXMCVSI','LBXRDW','ferritin','stfr','body_iron','serum_iron']; GLY=['LBXGH','LBXGLU','LBXIN','homa']
S=[run_seg_table('S1_cycles_I+P_only(2015-Mar2020)',PRIM+EXT,only_cycles=['I','P']),
   run_seg_table('S2_excl_iron_supplement_users',[o for o in PRIM if o[0] in IRON],extra_mask=lambda d:d.supp_iron==0),
   run_seg_table('S2_excl_folate_supplement_users',[o for o in PRIM if o[0]=='rbc_folate'],extra_mask=lambda d:d.supp_fol==0),
   run_seg_table('S3_complete_case_continuous_PIR',PRIM+EXT,pir='cont'),
   run_seg_table('S4_excl_RHD143_unknown_pregnancy',PRIM+EXT,extra_mask=lambda d:d.RHD143!=9),
   run_seg_table('S5_excl_CRP>5mg/L',[o for o in PRIM if o[0] in ('ferritin','body_iron','stfr')],extra_mask=lambda d:d.crp<=5),
   run_seg_table('S6_excl_self-reported_diabetes',[o for o in OUT if o[0] in GLY],extra_mask=lambda d:d.DIQ010!=1)]
S=pd.concat(S); S.to_csv(O+'T8_sensitivity_segment_contrasts.csv',index=False)
print(T.round(3).to_string()); print(G.round(3).to_string())
