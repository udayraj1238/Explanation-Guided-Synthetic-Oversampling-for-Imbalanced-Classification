import os
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import warnings
warnings.filterwarnings('ignore')

def compute_rank_biserial(diffs):
    non_zero = diffs[diffs != 0]
    n = len(non_zero)
    if n == 0: return 0.0
    ranks = rankdata(np.abs(non_zero))
    w_plus = np.sum(ranks[non_zero > 0])
    w_minus = np.sum(ranks[non_zero < 0])
    return float((w_plus - w_minus) / (w_plus + w_minus)) if (w_plus + w_minus) > 0 else 0.0

def run_statistical_tests():
    os.makedirs('results', exist_ok=True)
    
    # 1. Main Grid (enlarged with svmsmote, smotetomek)
    df = pd.read_csv('results/raw_results.csv')
    metrics = ['f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds']
    group_cols = ['dataset', 'classifier', 'resampler']
    
    summary_mean = df.groupby(group_cols)[metrics].mean().reset_index()
    summary_std = df.groupby(group_cols)[metrics].std().reset_index()
    
    summary_table = summary_mean.copy()
    for m in metrics:
        summary_table[m] = summary_mean[m].round(3).astype(str) + " ± " + summary_std[m].round(3).astype(str)
    summary_table.to_csv('results/summary_table.csv', index=False)
    
    stats_results = {
        "friedman": {},
        "wilcoxon_acsmote_vs_baseline": {},
        "cost_sensitive_wilcoxon": {}
    }
    
    classifiers = df['classifier'].unique()
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'svmsmote', 'smotetomek', 'acsmote']
    
    for clf in classifiers:
        stats_results["friedman"][clf] = {}
        stats_results["wilcoxon_acsmote_vs_baseline"][clf] = {}
        clf_data = summary_mean[summary_mean['classifier'] == clf]
        
        for metric in [m for m in metrics if m != 'fit_time_seconds']:
            friedman_data = []
            for res in resamplers:
                vals = clf_data[clf_data['resampler'] == res].sort_values('dataset')[metric].values
                friedman_data.append(vals)
                
            try:
                stat, p = friedmanchisquare(*friedman_data)
                stats_results["friedman"][clf][metric] = {"stat": stat, "p": p}
            except ValueError:
                stats_results["friedman"][clf][metric] = {"stat": None, "p": None}
                
        for baseline in resamplers:
            if baseline == 'acsmote': continue
            stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline] = {}
            for metric in [m for m in metrics if m != 'fit_time_seconds']:
                acsmote_vals = clf_data[clf_data['resampler'] == 'acsmote'].sort_values('dataset')[metric].values
                baseline_vals = clf_data[clf_data['resampler'] == baseline].sort_values('dataset')[metric].values
                
                if len(acsmote_vals) == len(baseline_vals) and len(acsmote_vals) > 0:
                    diffs = acsmote_vals - baseline_vals
                    wins = int(np.sum(diffs > 0))
                    losses = int(np.sum(diffs < 0))
                    ties = int(np.sum(diffs == 0))
                    r_es = compute_rank_biserial(diffs)
                    median_diff = float(np.median(diffs))
                    
                    if np.all(diffs == 0):
                        stat, p = 0.0, 1.0
                    else:
                        stat, p = wilcoxon(acsmote_vals, baseline_vals)
                    
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric] = {
                        "stat": float(stat) if not np.isnan(stat) else None,
                        "p": float(p) if not np.isnan(p) else None,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "r_rank_biserial": r_es,
                        "median_diff": median_diff
                    }

    # 2. Cost-Sensitive Analysis
    if os.path.exists('results/costsensitive_results.csv'):
        df_cs = pd.read_csv('results/costsensitive_results.csv')
        cs_mean = df_cs.groupby(['dataset', 'classifier'])[metrics].mean().reset_index()
        for clf in classifiers:
            stats_results["cost_sensitive_wilcoxon"][clf] = {}
            clf_cs_data = cs_mean[cs_mean['classifier'] == clf]
            clf_ac_data = summary_mean[(summary_mean['classifier'] == clf) & (summary_mean['resampler'] == 'acsmote')]
            for metric in [m for m in metrics if m != 'fit_time_seconds']:
                ac_vals = clf_ac_data.sort_values('dataset')[metric].values
                cs_vals = clf_cs_data.sort_values('dataset')[metric].values
                if len(ac_vals) == len(cs_vals) and len(ac_vals) > 0:
                    diffs = ac_vals - cs_vals
                    r_es = compute_rank_biserial(diffs)
                    stat, p = wilcoxon(ac_vals, cs_vals) if not np.all(diffs == 0) else (0.0, 1.0)
                    stats_results["cost_sensitive_wilcoxon"][clf][metric] = {
                        "p": float(p) if not np.isnan(p) else None,
                        "wins_for_acsmote": int(np.sum(diffs > 0)),
                        "losses_for_acsmote": int(np.sum(diffs < 0)),
                        "r_rank_biserial": r_es
                    }

    # 3. Multiseed Analysis (Consistency Check)
    if os.path.exists('results/multiseed_results.csv'):
        df_ms = pd.read_csv('results/multiseed_results.csv')
        # Combine raw and multiseed for seeds 42, 123, 2026
        df_seed42 = df[df['resampler'].isin(['none', 'smote', 'bsmote', 'adasyn', 'acsmote'])].copy()
        df_seed42['seed'] = 42
        df_all = pd.concat([df_seed42, df_ms], ignore_index=True)
        ms_mean = df_all.groupby(['dataset', 'classifier', 'resampler', 'seed'])[metrics].mean().reset_index()
        
        stats_results["multiseed_consistency"] = {}
        for clf in classifiers:
            stats_results["multiseed_consistency"][clf] = {}
            for metric in [m for m in metrics if m != 'fit_time_seconds']:
                stats_results["multiseed_consistency"][clf][metric] = {}
                for baseline in ['none', 'smote', 'bsmote', 'adasyn']:
                    stats_results["multiseed_consistency"][clf][metric][baseline] = {}
                    for seed in [42, 123, 2026]:
                        ac_vals = ms_mean[(ms_mean['classifier'] == clf) & (ms_mean['resampler'] == 'acsmote') & (ms_mean['seed'] == seed)].sort_values('dataset')[metric].values
                        b_vals = ms_mean[(ms_mean['classifier'] == clf) & (ms_mean['resampler'] == baseline) & (ms_mean['seed'] == seed)].sort_values('dataset')[metric].values
                        if len(ac_vals) == len(b_vals) and len(ac_vals) > 0:
                            diffs = ac_vals - b_vals
                            stats_results["multiseed_consistency"][clf][metric][baseline][f"seed_{seed}"] = {
                                "wins": int(np.sum(diffs > 0)),
                                "losses": int(np.sum(diffs < 0))
                            }

    with open('results/statistical_tests.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    print("Saved results/statistical_tests.json with effect sizes and robust scaling.")

if __name__ == '__main__':
    run_statistical_tests()
