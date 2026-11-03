"""Export the full analytic file (ALL DEMO rows; design intact) for independent R survey re-analysis.
Weights are NOT pre-computed here: R rebuilds the combined weights from raw 2-yr / prepandemic weights."""
import pandas as pd, numpy as np
B='./'
A=pd.read_pickle(B+'data/step1_analytic.pkl').copy()
A.loc[A.smk=='miss','smk']='never'; A.loc[A.parity=='miss','parity']='1'; A.loc[A.educ=='miss','educ']='>HS'
cols=['SEQN','cycle','SDMVSTRA','SDMVPSU','STR','PSU','WTMEC','WTINT','WTSAF2YR','WTSA2YR','WTFOLPRP','WTSH2YR','WTDRD1',
      'grp','seg','month','bf','gdm','RIDAGEYR','race','educ','pircat','parity','smk','married','INDFMPIR','RHD143','DIQ010',
      'ferritin','stfr','body_iron','LBXFER','LBXTFR','rbc_folate','LBXRBFSI','LBDRFOSI','homa','LBXGLU','LBXIN','LBXHGB','crp','BMXWAIST','BMXBMI',
      'sleep','supp_iron','supp_fol','DSD010','DSQTIRON','DSQTFDFE','DSQTFA','DR1TIRON','DR1TFDFE','DR1TKCAL','DR1DRSTZ','LBXVIDMS','phq9']
A[[c for c in cols if c in A]].to_csv(B+'data/derived/analytic_for_R.csv.gz',index=False)
print(A.shape, A.grp.value_counts().to_dict())
