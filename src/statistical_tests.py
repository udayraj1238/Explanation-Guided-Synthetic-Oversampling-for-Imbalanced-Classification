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

    # 4. Multiple Comparisons Correction (Main Grid)
    from statsmodels.stats.multitest import multipletests
    for clf in classifiers:
        for metric in [m for m in metrics if m != 'fit_time_seconds']:
            # Collect p-values for this family (7 baselines)
            p_vals = []
            baselines_in_family = []
            for baseline in resamplers:
                if baseline == 'acsmote': continue
                if baseline in stats_results["wilcoxon_acsmote_vs_baseline"][clf] and metric in stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline]:
                    p = stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["p"]
                    if p is not None:
                        p_vals.append(p)
                        baselines_in_family.append(baseline)
            
            if len(p_vals) > 0:
                reject_holm, p_holm, _, _ = multipletests(p_vals, method='holm')
                reject_bh, p_bh, _, _ = multipletests(p_vals, method='fdr_bh')
                for i, baseline in enumerate(baselines_in_family):
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["p_corr_holm"] = float(p_holm[i])
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["significant_holm"] = bool(reject_holm[i])
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["p_corr_fdr_bh"] = float(p_bh[i])
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["significant_fdr_bh"] = bool(reject_bh[i])
                    # Keep legacy 'p_corr' for compatibility if needed
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["p_corr"] = float(p_holm[i])
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric]["significant_after_correction"] = bool(reject_holm[i])

    # 4.5 Multiple Comparisons Correction (Cost-Sensitive)
    if "cost_sensitive_wilcoxon" in stats_results:
        for clf in classifiers:
            p_vals = []
            metrics_in_family = []
            if clf in stats_results["cost_sensitive_wilcoxon"]:
                for metric in [m for m in metrics if m != 'fit_time_seconds']:
                    if metric in stats_results["cost_sensitive_wilcoxon"][clf]:
                        p = stats_results["cost_sensitive_wilcoxon"][clf][metric].get("p")
                        if p is not None:
                            p_vals.append(p)
                            metrics_in_family.append(metric)
                
                if len(p_vals) > 0:
                    _, p_holm, _, _ = multipletests(p_vals, method='holm')
                    _, p_bh, _, _ = multipletests(p_vals, method='fdr_bh')
                    for i, metric in enumerate(metrics_in_family):
                        stats_results["cost_sensitive_wilcoxon"][clf][metric]["p_corr_holm"] = float(p_holm[i])
                        stats_results["cost_sensitive_wilcoxon"][clf][metric]["p_corr_fdr_bh"] = float(p_bh[i])

    # 5. Attribution Ablation Significance (Pooled and abalone_19-specific)
    if os.path.exists('results/attribution_ablation_results.csv'):
        df_abl = pd.read_csv('results/attribution_ablation_results.csv')
        abl_mean = df_abl.groupby(['dataset', 'attribution_method'])['gmean'].mean().reset_index()
        
        shap_vals = abl_mean[abl_mean['attribution_method'] == 'shap'].sort_values('dataset')['gmean'].values
        unif_vals = abl_mean[abl_mean['attribution_method'] == 'uniform'].sort_values('dataset')['gmean'].values
        
        if len(shap_vals) == len(unif_vals) and len(shap_vals) > 0:
            stat_pooled, p_pooled = wilcoxon(shap_vals, unif_vals)
        else:
            stat_pooled, p_pooled = None, None
            
        df_aba = df_abl[df_abl['dataset'] == 'abalone_19']
        shap_folds = df_aba[df_aba['attribution_method'] == 'shap'].sort_values('fold_id')['gmean'].values
        unif_folds = df_aba[df_aba['attribution_method'] == 'uniform'].sort_values('fold_id')['gmean'].values
        
        if len(shap_folds) == len(unif_folds) and len(shap_folds) > 0:
            stat_aba, p_aba = wilcoxon(shap_folds, unif_folds)
        else:
            stat_aba, p_aba = None, None
            
        stats_results["attribution_ablation"] = {
            "pooled_n5": {
                "p": float(p_pooled) if p_pooled is not None else None,
                "note": "n=5 limits statistical power, interpret with caution"
            },
            "abalone_19_folds_n15": {
                "p": float(p_aba) if p_aba is not None else None
            }
        }

    with open('results/statistical_tests.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    print("Saved results/statistical_tests.json with Holm-Bonferroni correction and ablation significance tests.")

if __name__ == '__main__':
    run_statistical_tests()
