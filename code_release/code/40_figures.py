"""Publication figures (English labels, 300-dpi PNG + vector PDF) -> output/figures/.
Inputs: output/step1/T*.csv, output/deep/*.csv. Curves displayed to month 22 (tail beyond 22 = sparse-n artefact; agreed)."""
import pandas as pd, numpy as np, matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B='./output/'; S=B+'step1/'; D=B+'deep/'; F=B+'figures/'
import os; os.makedirs(F,exist_ok=True)
mpl.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.6})
C0='#1f4e79'; CB='#c55a11'; CG='#7f7f7f'; SEGC={'0-6':'#c00000','7-12':'#ed7d31','13-24':'#2e75b6'}; MAXM=22
T2=pd.read_csv(S+'T2_segment_vs_control_adjusted.csv'); T3=pd.read_csv(S+'T3a_rcs_curve_diff_vs_control.csv'); T3b=pd.read_csv(S+'T3b_rcs_tests.csv')
T5=pd.read_csv(S+'T5_binary_prevalence_OR.csv'); T6=pd.read_csv(S+'T6_breastfeeding_0to12mo.csv'); T7=pd.read_csv(S+'T7_gdm_exploratory_interaction.csv')
T8=pd.read_csv(S+'T8_sensitivity_segment_contrasts.csv'); T0=pd.read_csv(S+'T0_n_by_month_cycle.csv')
LAB={'LBXRDW':'RDW (%)','ferritin':'Ferritin','body_iron':'Body iron (mg/kg)','stfr':'sTfR','rbc_folate':'RBC folate','LBXTC':'Total cholesterol (mg/dL)',
     'LBXSATSI':'ALT','LBXSUA':'Uric acid (mg/dL)','uacr':'UACR','homa':'HOMA-IR (fasting)','LBXHGB':'Hemoglobin (g/dL)','LBXMCVSI':'MCV (fL)',
     'LBXWBCSI':'WBC (10^9/L)','LBXPLTSI':'Platelets (10^9/L)','serum_iron':'Serum iron (µg/dL)','LBXGH':'HbA1c (%)','LBDHDD':'HDL-C (mg/dL)',
     'LBXSASSI':'AST','LBXSGTSI':'GGT','LBXSAL':'Albumin (g/dL)','LBXSCR':'Creatinine (mg/dL)','egfr':'eGFR (mL/min/1.73m²)','crp':'CRP',
     'sbp':'SBP (mmHg)','dbp':'DBP (mmHg)','BMXBMI':'BMI (kg/m²)','LBXTR':'Triglycerides (fasting)','LBDLDL':'LDL-C (mg/dL, fasting)',
     'LBXGLU':'Fasting glucose (mg/dL)','LBXIN':'Fasting insulin','URXUIO':'Urinary iodine','LBXVIDMS':'25(OH)D (nmol/L)','sleep':'Sleep (h/night)',
     'phq9':'PHQ-9 score','BMXWAIST':'Waist circumference (cm)','LBXBPB':'Blood lead','LBXBCD':'Blood cadmium','LBXTHG':'Blood total mercury',
     'DR1TIRON':'Dietary iron (day 1)','DR1TFDFE':'Dietary folate DFE (day 1)','DR1TKCAL':'Energy intake (kcal/d)'}
def save(fig,name):
    fig.savefig(F+name+'.png',dpi=300,bbox_inches='tight'); fig.savefig(F+name+'.pdf',bbox_inches='tight'); plt.close(fig)
def curve(ax,var,title=None,highlight=False,letter=None):
    c=T3[(T3['var']==var)&(T3.month<=MAXM)]; sc=c.scale.iloc[0]
    ax.fill_between(c.month,c.lo,c.hi,color=C0,alpha=.18,lw=0); ax.plot(c.month,c.est,color=C0,lw=1.3)
    ax.axhline(0,color='k',lw=.6,ls='--')
    t=T2[T2['var']==var]
    for _,r in t.iterrows():
        x={'0-6':3.5,'7-12':9.5,'13-24':17.5}[r.segment]
        ax.errorbar(x,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o',ms=3.2,color=SEGC[r.segment],elinewidth=.9,capsize=1.8,zorder=5)
    s=T3b[T3b['var']==var].iloc[0]; q=s.q_overall_FDR
    qs='q<0.001' if q<0.001 else f'q={q:.3f}'
    ax.text(.97,.95,f'Overall {qs}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
    ax.set_xlim(.5,MAXM+.5); ax.set_xticks([1,6,12,18,22])
    ax.set_title(title or LAB.get(var,var),fontweight='bold' if highlight else None,color='#7b2c00' if highlight else 'k')
    ax.set_ylabel('Difference vs comparator (%)' if sc=='% diff' else 'Difference vs comparator')
    if highlight:
        for sp in ['left','bottom']: ax.spines[sp].set_color('#7b2c00'); ax.spines[sp].set_linewidth(1.1)
    if letter: ax.text(-.22,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)

# ---------------- Figure 1: flow + n by month x cycle ----------------
fig=plt.figure(figsize=(7.2,3.6)); ax=fig.add_axes([0,0,.46,1]); ax.axis('off')
boxes=[(0.5,0.93,'NHANES 2007–2008 to 2017–Mar 2020 (6 cycles)\nwomen aged 20–44 y'),
       (0.5,0.73,'Reported months since last delivery\n(RHQ197) 1–24: n = 933'),
       (0.5,0.47,'Postpartum analytic sample\nn = 860 (all with MEC exam)\n0–6 mo: 265 | 7–12 mo: 263 | 13–24 mo: 332'),
       (0.5,0.13,'Comparator (same age, not pregnant;\nnulliparous or last birth ≥3 y): n = 4,959')]
for x,y,t in boxes:
    ax.add_patch(FancyBboxPatch((x-.42,y-.075),.84,.15,boxstyle='round,pad=0.01',fc='#eef3f8',ec=C0,lw=.8,transform=ax.transAxes))
    ax.text(x,y,t,ha='center',va='center',fontsize=7,transform=ax.transAxes)
ax.annotate('',xy=(.5,.81),xytext=(.5,.855),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=.8))
ax.annotate('',xy=(.5,.555),xytext=(.5,.655),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=.8))
ax.text(.98,.605,'Excluded: currently pregnant\n(RHD143 = 1 or RIDEXPRG = 1), n = 73',fontsize=6.3,ha='right',va='center',transform=ax.transAxes,color='#7b2c00',bbox=dict(fc='white',ec='none',pad=1))
ax.text(.5,.30,'Subsamples: iron panel (2007–10, 2015–Mar 2020) n = 599–602;\nfasting n = 396; urinary iodine n = 364; 25(OH)D n = 734; sleep n = 856; PHQ-9 n = 859',fontsize=6.2,ha='center',va='center',transform=ax.transAxes,style='italic')
ax.text(0,1.0,'A',fontweight='bold',fontsize=10,transform=ax.transAxes)
ax2=fig.add_axes([.55,.12,.44,.8]); cyc=['E','F','G','H','I','P']; cl={'E':'2007–08','F':'2009–10','G':'2011–12','H':'2013–14','I':'2015–16','P':'2017–Mar 2020'}
cols=plt.cm.Blues(np.linspace(.35,.95,6)); bot=np.zeros(len(T0))
for c,co in zip(cyc,cols): ax2.bar(T0.month,T0[c],bottom=bot,color=co,width=.85,label=cl[c],lw=0); bot+=T0[c].values
ax2.set_xlabel('Months since delivery'); ax2.set_ylabel('Unweighted n'); ax2.set_xticks([1,6,12,18,24]); ax2.legend(frameon=False,ncol=2,fontsize=6,loc='upper right')
ax2.axvspan(22.5,24.5,color='grey',alpha=.12,lw=0); ax2.text(23.5,ax2.get_ylim()[1]*.6,'not\nshown in\ncurves',fontsize=5.5,ha='center',color='grey')
ax2.text(-.15,1.04,'B',fontweight='bold',fontsize=10,transform=ax2.transAxes)
save(fig,'Fig1_flow_and_n_by_month_cycle')

# ---------------- Figure 2: main spline panel (9) ----------------
MAIN9=['body_iron','ferritin','stfr','rbc_folate','LBXRDW','LBXTC','LBXSATSI','uacr','homa','LBXSUA']
# agreed 9 panels: RDW, ferritin, sTfR/body iron, RBC folate, TC, ALT, uric acid, UACR, HOMA-IR -> body iron is the iron-store panel (sTfR in Fig S6)
MAIN9=['body_iron','ferritin','rbc_folate','LBXRDW','LBXTC','LBXSATSI','LBXSUA','uacr','homa']
fig,axs=plt.subplots(3,3,figsize=(7.2,6.6)); 
for i,(ax,v) in enumerate(zip(axs.flat,MAIN9)):
    curve(ax,v,highlight=v in('body_iron',),letter='ABCDEFGHI'[i])
    if i>=6: ax.set_xlabel('Months since delivery')
from matplotlib.lines import Line2D
h=[Line2D([],[],color=C0,lw=1.3,label='Restricted cubic spline (95% CI)')]+[Line2D([],[],marker='o',ls='',color=SEGC[s],label=f'{s} mo segment (95% CI)') for s in SEGC]
fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=(0,.03,1,1)); save(fig,'Fig2_main_spline_difference_curves')

# ---------------- Figure 3: recovery time bars ----------------
R=[]
for v in T3['var'].unique():
    c=T3[(T3['var']==v)&(T3.month<=MAXM)].sort_values('month'); sig=c.sig.values.astype(bool)
    last=int(c.month[sig].max()) if sig.any() else 0
    s=T3b[T3b['var']==v].iloc[0]
    R.append(dict(var=v,last=last,q=s.q_overall_FDR,tier=s.tier))
R=pd.DataFrame(R); R=R[(R.q<0.05)&(R['last']>0)&(~R.tier.isin(['extended_diet','appendix_metal']))].sort_values('last')
fig,ax=plt.subplots(figsize=(4.6,4.2))
col=[('#7b2c00' if v in('body_iron','ferritin','stfr','LBXMCVSI') else '#2e75b6' if v=='rbc_folate' else '#548235' if v in ('BMXWAIST','LBXVIDMS','LBXSUA','LBDHDD') else CG) for v in R['var']]
ax.barh([LAB.get(v,v) for v in R['var']],R['last'],color=col,height=.65)
for y,(l,q) in enumerate(zip(R['last'],R.q)): ax.text(l+.25,y,f'{l}',va='center',fontsize=6.5)
ax.axvline(6,color='k',ls=':',lw=.7); ax.text(6.2,-.9,'6 months',fontsize=6.5)
ax.set_xlabel('Last month at which the 95% CI excludes zero (spline, months 1–22)'); ax.set_xlim(0,24)
h=[plt.Rectangle((0,0),1,1,color=c) for c in ['#7b2c00','#2e75b6','#548235',CG]]
ax.legend(h,['Iron status','RBC folate','Late / persistent','Other outcomes'],frameon=False,loc='lower right')
ax.text(0,-.16,'Only outcomes with family-wise FDR q<0.05 for the overall spline test. Sparse tail beyond 22 mo not evaluated.',transform=ax.transAxes,fontsize=5.8,color='grey')
save(fig,'Fig3_recovery_time')

# ---------------- Figure 4: binary prevalence ----------------
A2=pd.read_csv(D+'A2_iron_deficiency_prevalence.csv')
bi=A2[(A2.outcome.str.startswith('Iron deficiency: body'))]
rows=[]
for o in T5.outcome.unique():
    t=T5[T5.outcome==o]; rows.append(dict(outcome=o,ctrl=t.prev_ctrl_.iloc[0] if 'prev_ctrl_' in t else t['prev_ctrl_%'].iloc[0],
        **{s:(t[t.segment==s]['prev_seg_%'].iloc[0],t[t.segment==s]['se_%'].iloc[0],t[t.segment==s].adj_OR.iloc[0],t[t.segment==s].lo.iloc[0],t[t.segment==s].hi.iloc[0]) for s in SEGC}))
a=bi[bi.analysis=='all'].set_index('group'); o_=bi[bi.analysis=='adjusted OR vs control'].set_index('group')
rows.insert(2,dict(outcome='Iron deficiency body iron<0',ctrl=a.loc['control','est'],**{s:(a.loc[s,'est'],a.loc[s,'se'],o_.loc[s,'est'],o_.loc[s,'lo'],o_.loc[s,'hi']) for s in SEGC}))
nice={'Anemia Hb<12':'Anemia (Hb <12 g/dL)','Iron deficiency ferritin<15':'Ferritin <15 ng/mL','Iron deficiency body iron<0':'Body iron <0 mg/kg','RBC folate <748 nmol/L':'RBC folate <748 nmol/L',
      '25(OH)D <50 nmol/L':'25(OH)D <50 nmol/L','HbA1c >=5.7%':'HbA1c ≥5.7%','BMI>=30':'BMI ≥30 kg/m²','Waist>=88cm':'Waist ≥88 cm','PHQ-9>=10':'PHQ-9 ≥10','Sleep<7h':'Sleep <7 h'}
fig,axs=plt.subplots(2,5,figsize=(7.4,3.9),sharey=False)
for ax,r in zip(axs.flat,rows):
    xs=np.arange(4); vals=[r['ctrl']]+[r[s][0] for s in SEGC]; ses=[np.nan]+[r[s][1] for s in SEGC]
    ax.bar(xs,vals,color=[CG]+list(SEGC.values()),width=.7,yerr=[0]+[1.96*x for x in ses[1:]],error_kw=dict(lw=.7,capsize=1.5))
    for x,s in zip(xs[1:],SEGC):
        orr,lo,hi=r[s][2:]; star='*' if (lo>1 or hi<1) else ''
        ax.text(x,(vals[x]+1.96*ses[x])*1.02+1,f'{orr:.2f}{star}',ha='center',fontsize=5.5,rotation=90,va='bottom')
    ax.set_xticks(xs); ax.set_xticklabels(['Comp.','0–6','7–12','13–24'],fontsize=6); ax.set_title(nice.get(r['outcome'],r['outcome']),fontsize=7)
    ax.set_ylim(0,max(np.array(vals[1:])+1.96*np.array(ses[1:]))*1.45)
axs[0,0].set_ylabel('Weighted prevalence (%)'); axs[1,0].set_ylabel('Weighted prevalence (%)')
fig.text(.5,-.01,'Bars: survey-weighted prevalence (95% CI). Numbers: adjusted OR vs comparator (*95% CI excludes 1).',ha='center',fontsize=6.5)
fig.tight_layout(); save(fig,'Fig4_binary_prevalence')

# ---------------- Figure 5: breastfeeding (0-12 mo), incl. waist ----------------
C=pd.read_csv(D+'C_waist_by_breastfeeding_0to12mo.csv')
V5=[('Body iron mg/kg','Body iron (mg/kg)'),('Ferritin ng/mL (harmonised e601)','Ferritin (%)'),('sTfR mg/L','sTfR (%)'),('RBC folate nmol/L','RBC folate (%)'),
    ('Hb g/dL','Hemoglobin (g/dL)'),('BMI kg/m2','BMI (kg/m²)'),('Waist cm','Waist (cm)'),('WAISTBMI','Waist adjusted for BMI (cm)')]
fig,axs=plt.subplots(2,4,figsize=(7.4,4.0))
cells=[('0-6','BF'),('0-6','noBF'),('7-12','BF'),('7-12','noBF')]
for ax,(o,lab) in zip(axs.flat,V5):
    for k,(s,b) in enumerate(cells):
        if o=='WAISTBMI':
            r=C[(C.analysis.str.startswith('C2'))&(C.cell==f'{b} {s}')].iloc[0]
        else:
            r=T6[(T6.outcome==o)&(T6.segment==s)&(T6.contrast==f'{b} vs ctrl')].iloc[0]
        ax.errorbar(k,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o' if b=='BF' else 's',color=CB if b=='BF' else C0,ms=3.5,capsize=1.8,elinewidth=.9)
    ax.axhline(0,color='k',ls='--',lw=.6); ax.set_xticks(range(4)); ax.set_xticklabels(['BF\n0–6','no BF\n0–6','BF\n7–12','no BF\n7–12'],fontsize=6)
    ax.set_title(lab,fontsize=7.5,fontweight='bold' if 'aist' in lab or 'Body iron' in lab else None); ax.set_xlim(-.5,3.5)
axs[0,0].set_ylabel('Adjusted difference vs comparator'); axs[1,0].set_ylabel('Adjusted difference vs comparator')
fig.text(.5,-.01,'Current breastfeeding (BF, self-report) among women 0–12 months postpartum; descriptive, not causal. Points: adjusted difference (95% CI) vs comparator.',ha='center',fontsize=6.3)
fig.tight_layout(); save(fig,'Fig5_breastfeeding_stratified')

# ---------------- Supplementary ----------------
# S1 GDM (standardised t-like)
g=T7.copy(); g['z']=g.gdm_diff_at_12mo/((g.hi-g.lo)/3.92); g['zl']=g.z-1.96; g['zh']=g.z+1.96; g=g.iloc[::-1]
fig,ax=plt.subplots(figsize=(4.8,6.4)); y=np.arange(len(g))
ax.errorbar(g.z,y,xerr=1.96,fmt='o',color=C0,ms=3,elinewidth=.7,capsize=1.2); ax.axvline(0,color='k',lw=.6,ls='--')
ax.set_yticks(y); ax.set_yticklabels([f'{o}  [{e:.2f} ({l:.2f}, {h:.2f})]' for o,e,l,h in zip(g.outcome,g.gdm_diff_at_12mo,g.lo,g.hi)],fontsize=5.8)
ax.set_xlabel('Standardised GDM difference at 12 mo (estimate/SE, ±1.96)'); ax.set_title('Supplementary: history of GDM (exploratory, n_GDM≈104)\nAll joint tests FDR q≥0.08',fontsize=7.5)
save(fig,'FigS1_GDM_exploratory')
# S2 sensitivity
KEY=['body_iron','ferritin','stfr','rbc_folate','LBXRDW','LBXTC','LBXSATSI','uacr','LBXSUA','homa','BMXWAIST','LBXHGB']
an={'main':'Main','S1_cycles_I+P_only(2015-Mar2020)':'S1 2015–Mar2020 only','S2_excl_iron_supplement_users':'S2 excl. supplement users','S2_excl_folate_supplement_users':'S2 excl. supplement users',
    'S3_complete_case_continuous_PIR':'S3 continuous PIR','S4_excl_RHD143_unknown_pregnancy':'S4 excl. unknown pregnancy','S5_excl_CRP>5mg/L':'S5 excl. CRP>5 mg/L','S6_excl_self-reported_diabetes':'S6 excl. diabetes'}
T2m=T2.assign(analysis='main'); SS=pd.concat([T2m,T8]); SS['an']=SS.analysis.map(an)
order=list(dict.fromkeys(an.values()))
fig,axs=plt.subplots(3,4,figsize=(7.4,6.4))
for ax,v in zip(axs.flat,KEY):
    t=SS[SS['var']==v]
    for k,a in enumerate(order):
        for j,s in enumerate(['0-6','7-12']):
            r=t[(t.an==a)&(t.segment==s)]
            if len(r)==0: continue
            r=r.iloc[0]; ax.errorbar(r.est,k+(j-.5)*.3,xerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o',color=SEGC[s],ms=2.4,elinewidth=.7)
    ax.axvline(0,color='k',ls='--',lw=.6); ax.set_yticks(range(len(order))); ax.set_yticklabels(order if v in (KEY[0],KEY[4],KEY[8]) else [],fontsize=5.5); ax.invert_yaxis()
    ax.set_title(LAB.get(v,v)+(' (%)' if t.scale.iloc[0]=='% diff' else ''),fontsize=7)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=SEGC[s],label=f'{s} mo') for s in ['0-6','7-12']],loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,.03,1,1)); save(fig,'FigS2_sensitivity_forest')
def grid(vars_,name,title):
    n=len(vars_); nc=4 if n>6 else 3; nr=int(np.ceil(n/nc))
    fig,axs=plt.subplots(nr,nc,figsize=(7.4,2.15*nr)); axs=np.atleast_1d(axs).flat
    for ax,v in zip(axs,vars_): curve(ax,v); ax.set_xlabel('Months since delivery',fontsize=6.5)
    for ax in list(axs): ax.axis('off')
    fig.suptitle(title,fontsize=8.5); fig.tight_layout(); save(fig,name)
grid(['LBXTR','LBDLDL','LBXGLU','LBXIN','homa','URXUIO','LBXVIDMS'],'FigS3_fasting_iodine_vitD_splines','Supplementary: fasting subsample, urinary iodine and 25(OH)D (3-knot splines; subsample weights >1e-6)')
grid(['LBXBPB','LBXBCD','LBXTHG'],'FigS4_metals_appendix','Supplementary (appendix): blood metals (exploratory)')
neg=[v for v in T3b[T3b.tier.isin(['primary','extended'])]['var'] if v not in MAIN9]
grid(neg,'FigS5_all_other_outcomes','Supplementary: all other primary and extended outcomes (including null results)')
# S6 iron deep dive
A1=pd.read_csv(D+'A1_iron_store_segment_contrasts.csv'); A3=pd.read_csv(D+'A3_iron_by_breastfeeding_0to12mo.csv')
fig,axs=plt.subplots(1,4,figsize=(7.6,3.2),gridspec_kw=dict(width_ratios=[1,1,1,1.05]))
anl=list(dict.fromkeys(A1.analysis)); short=['Main','Excl. iron suppl.','Excl. CRP>5','Excl. both','Diet sample (DR1 wt)','+ dietary iron','+ suppl. covariate']
for ax,o,lab in zip(axs[:3],['Body iron (mg/kg)','Ferritin','sTfR'],['Body iron (mg/kg)','Ferritin (%)','sTfR (%)']):
    t=A1[A1.outcome==o]
    for k,a in enumerate(anl):
        for j,s in enumerate(SEGC):
            r=t[(t.analysis==a)&(t.cell==s)].iloc[0]; ax.errorbar(r.est,k+(j-1)*.22,xerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o',color=SEGC[s],ms=2.3,elinewidth=.7)
    ax.axvline(0,color='k',ls='--',lw=.6); ax.set_yticks(range(len(anl))); ax.set_yticklabels(short if o.startswith('Body') else [],fontsize=6); ax.invert_yaxis(); ax.set_title(lab,fontsize=7.5)
ax=axs[3]; p=A2[A2.analysis=='all']
outs=['Iron deficiency: body iron <0 mg/kg','Iron deficiency WITHOUT anemia (body iron<0 & Hb>=12)','Iron-deficiency anemia (body iron<0 & Hb<12)','Ferritin <30 ng/mL']
labs=['Body iron <0','ID without anemia','ID anemia','Ferritin <30']
w=.2
for j,g_ in enumerate(['control','0-6','7-12','13-24']):
    vals=[p[(p.outcome==o)&(p.group==g_)].est.iloc[0] for o in outs]; ses=[p[(p.outcome==o)&(p.group==g_)].se.iloc[0] for o in outs]
    ax.bar(np.arange(4)+(j-1.5)*w,vals,w,yerr=1.96*np.array(ses),color=[CG,*SEGC.values()][j],error_kw=dict(lw=.5),label=['Comparator','0–6 mo','7–12 mo','13–24 mo'][j])
ax.set_xticks(range(4)); ax.set_xticklabels(labs,rotation=30,ha='right',fontsize=6); ax.set_ylabel('Weighted prevalence (%)'); ax.legend(frameon=False,fontsize=5.5); ax.set_title('Iron deficiency prevalence',fontsize=7.5)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=SEGC[s],label=f'{s} mo') for s in SEGC],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.38,-.04))
fig.tight_layout(rect=(0,.04,1,1)); save(fig,'FigS6_deep_iron_store_package')
# S7 folate deep dive
Bf=pd.read_csv(D+'B_folate_mechanism.csv')
fig,axs=plt.subplots(1,3,figsize=(7.4,2.9))
ax=axs[0]; m=Bf[Bf.analysis.str.startswith('B1 weighted mean')]
for st,co,mk in [('supplement users (folate/folic acid, 30-d DSQ)',CB,'o'),('non-users',C0,'s')]:
    t=m[m.stratum==st].set_index('cell').loc[['control','0-6','7-12','13-24']]
    ax.errorbar(range(4),t.est,yerr=1.96*t.se,fmt=mk+'-',color=co,ms=3.5,capsize=2,lw=.9,label='Supplement users' if co==CB else 'Non-users')
ax.set_xticks(range(4)); ax.set_xticklabels(['Comp.','0–6','7–12','13–24']); ax.set_ylabel('RBC folate (ng/mL), weighted mean'); ax.legend(frameon=False); ax.set_title('By folate-supplement use (30-day)',fontsize=7.5)
pint=Bf[Bf.analysis.str.startswith('B2')].p.iloc[0]; ax.text(.03,.03,f'Supplement × segment p={pint:.2f}',transform=ax.transAxes,fontsize=6)
ax=axs[1]; mods=['B4a intake sample, unadjusted (DR1 wt)','B4b + log dietary DFE','B4c + log total DFE (diet+supplement)','B4d + log total DFE + supplement use','B5 cycles I+P only (2015-Mar2020)']
ml=['Intake sample','+ dietary DFE','+ total DFE','+ total DFE + suppl.','2015–Mar2020 only']
for k,a in enumerate(mods):
    for j,s in enumerate(SEGC):
        r=Bf[(Bf.analysis==a)&(Bf.cell==s)].iloc[0]; ax.errorbar(r.est,k+(j-1)*.22,xerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o',color=SEGC[s],ms=2.6,elinewidth=.7)
ax.axvline(0,color='k',ls='--',lw=.6); ax.set_yticks(range(5)); ax.set_yticklabels(ml,fontsize=6); ax.invert_yaxis(); ax.set_xlabel('Adjusted % difference vs comparator'); ax.set_title('Intake adjustment',fontsize=7.5)
ax=axs[2]; v=Bf[Bf.analysis.str.startswith('B6')]
ax.bar(range(len(v)),v.est,yerr=1.96*v.se,color=[CG,'#7030a0']+list(SEGC.values()),width=.65,error_kw=dict(lw=.6))
ax.set_xticks(range(len(v))); ax.set_xticklabels(['Comp.','1–12','1–6','7–12','13–24'],fontsize=6.5); ax.set_ylabel('RBC folate (ng/mL)')
ax.axhline(606,color='#7030a0',ls=':',lw=.8); ax.text(-.4,720,'dotted line: Vladutiu 2021, 606 ng/mL (year 1)',fontsize=5.5,ha='left',color='#7030a0')
ax.set_title('2007–2016, comparable to PMID 33236942',fontsize=7.5); ax.set_ylim(0,760)
fig.tight_layout(); save(fig,'FigS7_deep_folate_mechanism')
# S8 waist x BF
fig,axs=plt.subplots(1,2,figsize=(6.4,2.8))
for ax,an_,tt in [(axs[0],'C1','Waist, adjusted (main covariates)'),(axs[1],'C2','Waist, additionally adjusted for BMI')]:
    t=C[(C.analysis.str.startswith(an_))&(C.contrast=='vs control')]
    for k,(s,b) in enumerate(cells):
        r=t[t.cell==f'{b} {s}'].iloc[0]; ax.errorbar(k,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o' if b=='BF' else 's',color=CB if b=='BF' else C0,ms=3.5,capsize=2)
    bb=C[(C.analysis.str.startswith(an_))&(C.contrast=='BF vs noBF')]
    ax.text(.02,.97,'BF − noBF: '+'; '.join(f'{r.cell} mo {r.est:.1f} ({r.lo:.1f}, {r.hi:.1f}) cm' for _,r in bb.iterrows()),transform=ax.transAxes,fontsize=5.6,va='top')
    ax.axhline(0,color='k',ls='--',lw=.6); ax.set_xticks(range(4)); ax.set_xticklabels(['BF\n0–6','no BF\n0–6','BF\n7–12','no BF\n7–12'],fontsize=6.5); ax.set_title(tt,fontsize=7.5); ax.set_ylabel('Difference vs comparator (cm)')
fig.tight_layout(); save(fig,'FigS8_deep_waist_by_breastfeeding')
print(sorted(os.listdir(F)))
