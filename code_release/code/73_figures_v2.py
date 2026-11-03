"""v2 figures (revision R1-R8). Re-uses code/40_figures.py verbatim except: output dir -> output/figures_v2/, display labels
'0–6' -> '1–6' (R8). Then overrides: Fig2 body-iron panel adds the simultaneous (sup-t) 95% band (R2); recovery figure
(manuscript Fig 4) redrawn with pointwise vs simultaneous significant months within 1–22 (R1/R2); binary-prevalence figure
(manuscript Fig 3) adds † for NCHS-unreliable cells (R6). Inputs: output/step1, output/deep, output/revision."""
import re
B='./'
src=open(B+'code/40_figures.py',encoding='utf-8').read()
src=src.replace("F=B+'figures/'","F=B+'figures_v2/'").replace('0–6','1–6')
src=src.replace("label=f'{s} mo segment (95% CI)'","label=f\"{s.replace('0-6','1-6')} mo segment (95% CI)\"").replace("label=f'{s} mo'","label=f\"{s.replace('0-6','1-6')} mo\"")
exec(compile(src,'40_figures_v2','exec'))
rows_b=list(rows)  # binary-figure rows built by 40_figures.py
RV='./output/revision/'
BAND=pd.read_csv(RV+'R2_simultaneous_bands_curves.csv'); SUM=pd.read_csv(RV+'R2_simultaneous_bands_summary.csv')
K=pd.read_csv(RV+'R1_spline_window_knot_sensitivity.csv'); REL=pd.read_csv(RV+'R6_nchs_reliability_flags.csv')
# ---- Fig 2 with simultaneous band on body iron ----
fig,axs=plt.subplots(3,3,figsize=(7.2,6.6))
for i,(ax,v) in enumerate(zip(axs.flat,MAIN9)):
    if v=='body_iron':
        b=BAND[BAND['var']==v]; ax.fill_between(b.month,b.lo_simul,b.hi_simul,color='#7b2c00',alpha=.10,lw=0)
        ax.plot(b.month,b.lo_simul,color='#7b2c00',lw=.6,ls=':'); ax.plot(b.month,b.hi_simul,color='#7b2c00',lw=.6,ls=':')
    curve(ax,v,highlight=v in('body_iron',),letter='ABCDEFGHI'[i])
    if i>=6: ax.set_xlabel('Months since delivery')
from matplotlib.lines import Line2D
h=[Line2D([],[],color=C0,lw=1.3,label='Spline (pointwise 95% CI)'),Line2D([],[],color='#7b2c00',lw=.8,ls=':',label='Simultaneous 95% band (A)')]+[Line2D([],[],marker='o',ls='',color=SEGC[s],label=f"{s.replace('0-6','1-6')} mo segment") for s in SEGC]
fig.legend(handles=h,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.02),fontsize=6.3)
fig.tight_layout(rect=(0,.03,1,1)); save(fig,'Fig2_main_spline_difference_curves')
# ---- recovery figure: month-by-month significance strip ----
q=T3b.set_index('var').q_overall_FDR; tier=T3b.set_index('var').tier
vs=[v for v in SUM['var'] if q[v]<0.05 and tier[v] not in ('extended_diet','appendix_metal')]
rows=[]
for v in vs:
    mk=K[(K['var']==v)&K.main_spec].iloc[0]; pat=mk.pattern
    s=SUM[SUM['var']==v].iloc[0]; sm=[] if s.simult_sig_months=='none' else [int(x) for x in str(s.simult_sig_months).split(',')]
    rng_=K[K['var']==v].initial_run_end; rows.append((v,pat,sm,int(mk.initial_run_end),int(rng_.min()),int(rng_.max())))
rows.sort(key=lambda r:(r[3],r[1].count('x')))
fig,ax=plt.subplots(figsize=(6.2,4.4))
for y,(v,pat,sm,ini,lo,hi) in enumerate(rows):
    for mth,ch in enumerate(pat,1):
        if ch=='x': ax.add_patch(plt.Rectangle((mth-.45,y-.32),.9,.64,color='#9dc3e6',lw=0))
        if mth in sm: ax.add_patch(plt.Rectangle((mth-.45,y-.32),.9,.64,color='#1f4e79',lw=0))
    ax.text(23.0,y,f'{ini}  ({lo}–{hi})',va='center',fontsize=6.3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([LAB.get(r[0],r[0]) for r in rows],fontsize=6.8)
ax.set_xlim(.4,26.5); ax.set_ylim(-.7,len(rows)-.3); ax.set_xticks([1,6,12,18,22]); ax.set_xlabel('Months since delivery (spline evaluated at months 1–22)')
ax.text(23.0,len(rows)-.2,'Initial run\n(3/4/5 knots)',fontsize=6,va='bottom')
ax.axvline(6.5,color='k',ls=':',lw=.6)
h=[plt.Rectangle((0,0),1,1,color='#9dc3e6'),plt.Rectangle((0,0),1,1,color='#1f4e79')]
ax.legend(h,['Pointwise 95% CI excludes 0','Simultaneous 95% band excludes 0'],frameon=False,loc='lower right',fontsize=6.3,bbox_to_anchor=(1,-.30),ncol=2)
save(fig,'Fig3_recovery_time')
# ---- binary prevalence with daggers ----
key={'Anemia Hb<12':'anemia_b','Iron deficiency ferritin<15':'iron_def_b','RBC folate <748 nmol/L':'rbcf_low_b','25(OH)D <50 nmol/L':'vitd_low_b','HbA1c >=5.7%':'a1c_pre_b',
     'BMI>=30':'obese_b','Waist>=88cm':'waist88_b','PHQ-9>=10':'phq10_b','Sleep<7h':'short_sleep_b','Iron deficiency body iron<0':'bi_neg'}
def dag(o,s):
    fam='Table2B_iron_cycles' if o=='Iron deficiency body iron<0' else 'Fig3_TableS3_all_cycles'
    r=REL[(REL.family==fam)&(REL.outcome==key[o])&(REL.cell==s)].iloc[0]
    return bool(r.flag=='suppress' or r.events<10 or r.rse>0.30)
fig,axs=plt.subplots(2,5,figsize=(7.4,3.9),sharey=False)
for ax,r in zip(axs.flat,rows_b):
    xs=np.arange(4); vals=[r['ctrl']]+[r[s][0] for s in SEGC]; ses=[np.nan]+[r[s][1] for s in SEGC]
    ax.bar(xs,vals,color=[CG]+list(SEGC.values()),width=.7,yerr=[0]+[1.96*x for x in ses[1:]],error_kw=dict(lw=.7,capsize=1.5))
    for x,s in zip(xs[1:],SEGC):
        orr,lo,hi=r[s][2:]; star='*' if (lo>1 or hi<1) else ''; d='†' if dag(r['outcome'],s) else ''
        ax.text(x,(vals[x]+1.96*ses[x])*1.02+1,f'{orr:.2f}{star}{d}',ha='center',fontsize=5.5,rotation=90,va='bottom')
    ax.set_xticks(xs); ax.set_xticklabels(['Comp.','1–6','7–12','13–24'],fontsize=6); ax.set_title(nice.get(r['outcome'],r['outcome']),fontsize=7)
    ax.set_ylim(0,max(np.array(vals[1:])+1.96*np.array(ses[1:]))*1.55)
axs[0,0].set_ylabel('Weighted prevalence (%)'); axs[1,0].set_ylabel('Weighted prevalence (%)')
fig.text(.5,-.03,'Bars: survey-weighted prevalence (95% CI). Numbers: adjusted OR vs comparator (*95% CI excludes 1).\n† Prevalence estimate unreliable by NCHS proportion standards (Korn–Graubard CI), RSE >0.30 or <10 cases.',ha='center',fontsize=6.3)
fig.tight_layout(); save(fig,'Fig4_binary_prevalence')
print('figures_v2 done')
