"""Supplementary Table S11: S1 (2015-March 2020) p values recomputed with residual df
(design df + 1 - number of model parameters = 19) instead of design df (40).
The Wald t statistic is recovered from the design-df p value and re-referred to t(resid_df)."""
import pandas as pd
from scipy import stats
O='./output/step1/'
t=pd.read_csv(O+'T8_sensitivity_segment_contrasts.csv')
s=t[t.analysis.str.startswith('S1_')][['analysis','outcome','segment','est','lo','hi','p','design_df']].copy()
s['resid_df']=19
tt=stats.t.isf(s.p/2,s.design_df)
s['p_resid']=2*stats.t.sf(tt,s.resid_df)
s.to_csv(O+'T8b_S1_residual_df_check.csv',index=False); print('T8b rows',len(s))
