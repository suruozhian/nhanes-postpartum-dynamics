"""v3 figures (EJN artwork rules). Re-runs code/73_figures_v2.py (which re-uses 40_figures.py) with these display-only changes:
output -> output/figures_v3/; 600-dpi RGB TIFF + vector PDF + EPS; width <=174 mm; lower-case panel letters; UK spelling
(anaemia, haemoglobin) to match the text; in-image footnotes removed (moved to captions); Fig 5 'women 0-12 months' note removed;
file names follow manuscript numbering (Fig3 = binary prevalence, Fig4 = significant-month strip). No data or estimate changes."""
import re, os, subprocess
B='./'
src=open(B+'code/40_figures.py',encoding='utf-8').read()
v2=open(B+'code/73_figures_v2.py',encoding='utf-8').read()
# 73 logic (substitutions to 40 source, then overrides)
src=src.replace("F=B+'figures/'","F=B+'figures_v3/'").replace('0–6','1–6')
src=src.replace("label=f'{s} mo segment (95% CI)'","label=f\"{s.replace('0-6','1-6')} mo segment (95% CI)\"").replace("label=f'{s} mo'","label=f\"{s.replace('0-6','1-6')} mo\"")
SAVE_OLD="fig.savefig(F+name+'.png',dpi=300,bbox_inches='tight'); fig.savefig(F+name+'.pdf',bbox_inches='tight'); plt.close(fig)"
assert SAVE_OLD in src
SAVE_NEW="""RENAME={'Fig1_flow_and_n_by_month_cycle':'Fig1','Fig2_main_spline_difference_curves':'Fig2','Fig4_binary_prevalence':'Fig3','Fig3_recovery_time':'Fig4','Fig5_breastfeeding_stratified':'Fig5'}
    nm=RENAME.get(name,name); nm=(nm.split('_')[0] if nm.startswith('FigS') else nm); W=174/25.4
    w,h=fig.get_size_inches()
    if w>W*0.97: fig.set_size_inches(W*0.97,h*W*0.97/w)
    fig.savefig(F+nm+'.pdf',bbox_inches='tight',pad_inches=0.02)
    fig.savefig(F+nm+'.tmp.png',dpi=600,bbox_inches='tight',pad_inches=0.02); plt.close(fig)
    from PIL import Image
    im=Image.open(F+nm+'.tmp.png').convert('RGB')
    if im.size[0]>int(W*600):
        im=im.resize((int(W*600),int(im.size[1]*W*600/im.size[0])),Image.LANCZOS)
    im.save(F+nm+'.tif',dpi=(600,600),compression='tiff_lzw'); im.save(F+nm+'.png',dpi=(600,600)); os.remove(F+nm+'.tmp.png')
    import subprocess; subprocess.run(['pdftops','-eps','-level3',F+nm+'.pdf',F+nm+'.eps'],check=True)"""
src=src.replace(SAVE_OLD,SAVE_NEW)
body=v2.split("exec(compile(src,'40_figures_v2','exec'))",1)[1]
full=src+"\n"+body
reps=[("letter='ABCDEFGHI'[i]","letter='abcdefghi'[i]"),
      ("ax.text(0,1.0,'A',","ax.text(0,1.0,'a',"),("ax2.text(-.15,1.04,'B',","ax2.text(-.15,1.04,'b',"),
      ("'Simultaneous 95% band (A)'","'Simultaneous 95% band (a)'"),
      ("'Hemoglobin (g/dL)'","'Haemoglobin (g/dL)'"),("'Anemia (Hb <12 g/dL)'","'Anaemia (Hb <12 g/dL)'"),
      ("labs=['Body iron <0','ID without anemia','ID anemia','Ferritin <30']","labs=['Body iron <0','ID without anaemia','ID anaemia','Ferritin <30']"),
      ("d='†' if dag(","d='ᵇ' if dag("),
      ("figsize=(7.2,6.6)","figsize=(6.8,6.3)"),("figsize=(7.4,3.9)","figsize=(6.8,3.7)"),("figsize=(7.4,4.0)","figsize=(6.8,3.8)"),("figsize=(7.2,3.6)","figsize=(6.8,3.5)"),
     ]
for a,b in reps:
    assert a in full,a; full=full.replace(a,b)
# remove in-image footnotes (now in captions)
for pat in [r"fig\.text\(\.5,-\.03,'Bars: survey-weighted prevalence.*?\n", r"fig\.text\(\.5,-\.01,'Current breastfeeding.*?\n", r"fig\.text\(\.5,-\.01,'Bars: survey-weighted prevalence.*?\n"]:
    full,n=re.subn(pat,"\n",full); assert n==1,pat
# no remaining forbidden labels in main-figure code
assert "0–12 months" not in full and "'0–6'" not in full
exec(compile(full,'76_figures_v3','exec'))
print('figures_v3 done', sorted(os.listdir(F)))
