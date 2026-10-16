import pandas as pd, numpy as np, re
from scipy.stats import wilcoxon, t as tdist, rankdata
import sys
R=sys.argv[1].rstrip('/')+'/'   # folder with raw_results.csv and lr_scaled_results.csv
OUT=sys.argv[2].rstrip('/')+'/'   # output folder
raw=pd.read_csv(R+'raw_results.csv',keep_default_na=False); ls=pd.read_csv(R+'lr_scaled_results.csv',keep_default_na=False)
mets=['f1','auc_pr','mcc','gmean','bal_acc']
mn={'f1':'F1','auc_pr':'AUC-PR','mcc':'MCC','gmean':'G-mean','bal_acc':'Balanced accuracy'}
base=['none','ros','smote','bsmote','adasyn','svmsmote','smotetomek']
bn={'none':'None','ros':'ROS','smote':'SMOTE','bsmote':'Borderline-SMOTE','adasyn':'ADASYN','svmsmote':'SVM-SMOTE','smotetomek':'SMOTE-Tomek'}
settings={'LR':raw[raw.classifier=='logreg'],'RF':raw[raw.classifier=='rf'],'XGB':raw[raw.classifier=='xgb'],'LR-std':ls}
def holm(p):
    p=np.array(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,idx in enumerate(o): run=max(run,(m-i)*p[idx]); adj[idx]=min(1,run)
    return adj
def bh(p):
    p=np.array(p); m=len(p); o=np.argsort(p)[::-1]; adj=np.empty(m); run=1
    for i,idx in enumerate(o): run=min(run,p[idx]*m/(m-i)); adj[idx]=run
    return adj
def rbc(d):
    nz=d[d!=0]
    if len(nz)==0: return 0.0
    r=rankdata(np.abs(nz)); wp=r[nz>0].sum(); wm=r[nz<0].sum(); return (wp-wm)/(wp+wm)
def wil(d): return 1.0 if np.all(d==0) else wilcoxon(d).pvalue
pm={s:g.groupby(['dataset','resampler'])[mets].mean() for s,g in settings.items()}
rows=[]
for s in settings:
    for met in mets:
        p=pm[s][met].unstack(); tmp=[]
        for b in base:
            d=(p['acsmote']-p[b]).values
            tmp.append(dict(setting=s,metric=mn[met],baseline=bn[b],wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),median_diff=float(np.median(d)),rank_biserial=rbc(d),p_raw=wil(d)))
        h=holm([r['p_raw'] for r in tmp]); b_=bh([r['p_raw'] for r in tmp])
        for r,x,y in zip(tmp,h,b_): r['p_holm']=x; r['p_bh']=y; rows.append(r)
full=pd.DataFrame(rows); full.to_csv(OUT+'pairwise_all_metrics.csv',index=False)
# verify Table III from paper
import os
t3=open('tables/tab_pairwise.tex').read() if os.path.exists('tables/tab_pairwise.tex') else ''
cells=re.findall(r'(\d+)/(\d+) \((1\.00|\.\d+)\)',t3)
exp=[]
for b in base:
    for met in ['G-mean','F1']:
        for s in ['LR','RF','XGB','LR-std']:
            r=full[(full.baseline==bn[b])&(full.metric==met)&(full.setting==s)].iloc[0]; exp.append((str(r.wins),str(r.losses),r.p_holm))
bad=0
for (w,l,p),(ew,el,ep) in zip(cells,exp):
    pv=1.0 if p=='1.00' else float(p)
    if not(w==ew and l==el and abs(pv-ep)<0.0015): bad+=1; print('MISMATCH',w,l,p,ew,el,round(ep,4))
print('Table III cells checked:',len(cells),'mismatches:',bad)
# global
p=full.p_raw.values; print('global holm min',holm(p).min(),'bh<.05',(bh(p)<.05).sum())
# equivalence
rng=np.random.default_rng(2026); eq=[]
for met in ['gmean','bal_acc']:
    sub=[]
    for s in settings:
        p_=pm[s][met].unstack(); d=(p_['acsmote']-p_['smote']).values; n=len(d)
        bs=np.array([np.median(rng.choice(d,n,replace=True)) for _ in range(20000)]); lo,hi=np.percentile(bs,[2.5,97.5])
        sd=d.std(ddof=1); se=sd/np.sqrt(n); mde=(tdist.ppf(.975,n-1)+tdist.ppf(.8,n-1))*se
        r=dict(setting=s,metric=mn[met],median_diff=np.median(d),ci_lo=lo,ci_hi=hi,mean_diff=d.mean(),sd_diff=sd,mde_80=mde)
        for dl in [0.01,0.02]:
            r[f'tost_p_{dl}']=max(wilcoxon(d+dl,alternative='greater').pvalue,wilcoxon(d-dl,alternative='less').pvalue)
        sub.append(r)
    for dl in [0.01,0.02]:
        h=holm([r[f'tost_p_{dl}'] for r in sub])
        for r,x in zip(sub,h): r[f'tost_p_holm_{dl}']=x
    eq+=sub
eqd=pd.DataFrame(eq); eqd.to_csv(OUT+'equivalence_tests.csv',index=False)
pd.set_option('display.width',250); print(eqd.round(4).to_string())
def fp(x): return '<0.001' if x<0.001 else f'{x:.3f}'
g=eqd[eqd.metric=='G-mean']
lines=[r'\begin{tabular}{@{}lccccc@{}}',r'\toprule',r'Setting & Median diff. & 95\% CI (median) & Min. det. diff. & TOST $p$ ($\pm0.01$) & TOST $p$ ($\pm0.02$) \\',r'\midrule']
for _,r in g.iterrows():
    lines.append(f"{r.setting} & ${r.median_diff:+.4f}$ & $[{r.ci_lo:+.4f},\\,{r.ci_hi:+.4f}]$ & {r.mde_80:.4f} & {fp(r['tost_p_holm_0.01'])} & {fp(r['tost_p_holm_0.02'])} \\\\")
lines+= [r'\bottomrule',r'\end{tabular}']
open(OUT+'tab_equiv_generated.tex','w').write('\n'.join(lines))
print('\n'.join(lines))
# supplementary latex
def fnum(x): return f'{x:.4f}'
tex=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=2cm]{geometry}',r'\usepackage{booktabs,longtable,array}',r'\usepackage[T1]{fontenc}',r'\setlength{\tabcolsep}{4pt}',r'\begin{document}',
r'\title{Supplementary Tables for ``Attribution-Constrained SMOTE: A Controlled Evaluation of Explanation-Guided Oversampling for Imbalanced Classification''}',r'\author{Uday Raj and Monika Sharma}\date{}\maketitle',
r'\noindent All tables are computed from the per-fold results released with the code (dataset-level means over 15 folds; $N=10$ datasets). AC-SMOTE is compared with each baseline by a two-sided exact Wilcoxon signed-rank test. Holm and Benjamini--Hochberg (BH) adjustments are applied within a family of seven comparisons (one setting, one metric). Median diff.\ is the median dataset-level difference (AC-SMOTE minus baseline); $r_{rb}$ is the matched-pairs rank-biserial correlation. Settings: LR = Logistic Regression on raw features; LR-std = standardized features (exploratory, seed 42).',
r'\section*{S1. Equivalence tests: AC-SMOTE versus SMOTE}',
r'\noindent TOST: two one-sided Wilcoxon signed-rank tests against margins $\pm\delta$; the reported $p$ is the larger of the two, Holm-adjusted across the four settings. CI: 95\% percentile bootstrap interval of the median (20\,000 resamples). Min.\ det.\ diff.: smallest difference detectable at 80\% power by a paired $t$-test with the observed standard deviation of the differences, $N=10$ (an approximation). Margins were chosen after the data were seen.',
r'\begin{center}\small\begin{tabular}{@{}llrrrrrrr@{}}\toprule Metric & Setting & Median & CI low & CI high & Min.\ det. & $p_{0.01}$ & $p_{0.02}$ & SD diff. \\ \midrule']
for _,r in eqd.iterrows():
    tex.append(f"{r.metric} & {r.setting} & {r.median_diff:+.4f} & {r.ci_lo:+.4f} & {r.ci_hi:+.4f} & {r.mde_80:.4f} & {fp(r['tost_p_holm_0.01'])} & {fp(r['tost_p_holm_0.02'])} & {r.sd_diff:.4f} \\\\")
tex+=[r'\bottomrule\end{tabular}\end{center}']
for k,met in enumerate(mets):
    tex.append(rf'\section*{{S{k+2}. Pairwise comparisons: {mn[met]}}}')
    tex.append(r'\begin{center}\footnotesize\begin{longtable}{@{}llcrrrrr@{}}\toprule Setting & Baseline & W/L/T & Median diff. & $p$ (raw) & $p$ (Holm) & $p$ (BH) & $r_{rb}$ \\ \midrule\endhead')
    for s in settings:
        sub=full[(full.metric==mn[met])&(full.setting==s)]
        for i,(_,r) in enumerate(sub.iterrows()):
            b='\\textbf{%s}'%fp(r.p_holm) if r.p_holm<0.05 else fp(r.p_holm)
            tex.append(f"{s if i==0 else ''} & {r.baseline} & {r.wins}/{r.losses}/{r.ties} & {r.median_diff:+.4f} & {fp(r.p_raw)} & {b} & {fp(r.p_bh)} & {r.rank_biserial:+.2f} \\\\")
        tex.append(r'\midrule')
    tex.append(r'\bottomrule\end{longtable}\end{center}')
tex.append(r'\noindent Bold: Holm-adjusted $p<0.05$ within its family of seven.')
tex.append(r'\end{document}')
open(OUT+'supplementary_tables.tex','w').write('\n'.join(tex))
