"""DRAFT internal diagnostic sketches only (not formal figures)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B='./'; O=B+'output/step1/'; D=B+'output/diagnostic/'
C=pd.read_csv(O+'T3a_rcs_curve_diff_vs_control.csv'); T2=pd.read_csv(O+'T2_segment_vs_control_adjusted.csv')
mid={'0-6':3.5,'7-12':9.5,'13-24':18.5}
for tiers,name in [(['primary'],'DRAFT_rcs_primary.png'),(['spline_only','extended','extended_diet','appendix_metal'],'DRAFT_rcs_secondary_extended_appendix.png')]:
    outs=C[C.tier.isin(tiers)].outcome.unique(); n=len(outs); nc=5; nr=int(np.ceil(n/nc))
    fig,ax=plt.subplots(nr,nc,figsize=(3.2*nc,2.6*nr)); ax=ax.ravel()
    for a,o in zip(ax,outs):
        c=C[C.outcome==o]; a.fill_between(c.month,c.lo,c.hi,alpha=.25); a.plot(c.month,c.est); a.axhline(0,c='k',lw=.6)
        s=T2[T2.outcome==o]; a.errorbar([mid[x] for x in s.segment],s.est,yerr=[s.est-s.lo,s.hi-s.est],fmt='o',c='C3',ms=3,capsize=2)
        a.set_title(o[:34]+(' [%]' if c.scale.iloc[0]=='% diff' else ''),fontsize=8); a.tick_params(labelsize=7)
    for a in ax[n:]: a.axis('off')
    fig.suptitle('DRAFT – internal diagnostic, not for publication: adjusted difference vs non-postpartum controls by months since delivery (RCS ±95%CI; red = 3-segment)',fontsize=9)
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(D+name,dpi=90); plt.close(fig)
# sample-size diagnostic by month and cycle
A=pd.read_pickle(B+'data/step1_analytic.pkl'); P=A[A.pp&(A.cycle!='J')]
ct=pd.crosstab(P.month,P.cycle); ct.to_csv(O+'T0_n_by_month_cycle.csv')
fig,a=plt.subplots(figsize=(7,3)); ct.plot.bar(stacked=True,ax=a,width=.9); a.set_title('DRAFT – n postpartum by RHQ197 month × cycle (E–H under-sample 13–24 mo by RHQ195 skip)',fontsize=8)
fig.tight_layout(); fig.savefig(D+'DRAFT_n_by_month_cycle.png',dpi=90)
