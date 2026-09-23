import os
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import warnings
warnings.filterwarnings('ignore')

def run_statistical_tests():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/raw_results.csv')
    
    # Calculate dataset-level means across the 15 folds
    metrics = ['f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds']
    group_cols = ['dataset', 'classifier', 'resampler']
    
    # For summary table, get mean and std
    summary_mean = df.groupby(group_cols)[metrics].mean().reset_index()
    summary_std = df.groupby(group_cols)[metrics].std().reset_index()
    
    # Combine into a mean ± std format for the paper
    summary_table = summary_mean.copy()
    for m in metrics:
        summary_table[m] = summary_mean[m].round(3).astype(str) + " ± " + summary_std[m].round(3).astype(str)
    summary_table.to_csv('results/summary_table.csv', index=False)
    print("Saved results/summary_table.csv")
    
    # Statistical tests dictionary
    stats_results = {
        "friedman": {},
        "wilcoxon_acsmote_vs_baseline": {}
    }
    
    classifiers = df['classifier'].unique()
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
    datasets = df['dataset'].unique()
    
    for clf in classifiers:
        stats_results["friedman"][clf] = {}
        stats_results["wilcoxon_acsmote_vs_baseline"][clf] = {}
        
        clf_data = summary_mean[summary_mean['classifier'] == clf]
        
        for metric in [m for m in metrics if m != 'fit_time_seconds']:
            # Friedman Test
            # Need a block per dataset, method as groups
            friedman_data = []
            for res in resamplers:
                # get metric values for this resampler across all 10 datasets, properly aligned
                vals = clf_data[clf_data['resampler'] == res].sort_values('dataset')[metric].values
                friedman_data.append(vals)
                
            try:
                # friedmanchisquare takes groups as positional arguments
                stat, p = friedmanchisquare(*friedman_data)
                stats_results["friedman"][clf][metric] = {"stat": stat, "p": p}
            except ValueError:
                stats_results["friedman"][clf][metric] = {"stat": None, "p": None}
                
        # Wilcoxon Signed-Rank Test: ACSMOTE vs baselines
        for baseline in resamplers:
            if baseline == 'acsmote':
                continue
                
            stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline] = {}
            
            for metric in [m for m in metrics if m != 'fit_time_seconds']:
                acsmote_vals = clf_data[clf_data['resampler'] == 'acsmote'].sort_values('dataset')[metric].values
                baseline_vals = clf_data[clf_data['resampler'] == baseline].sort_values('dataset')[metric].values
                
                # Wilcoxon signed-rank test
                if len(acsmote_vals) == len(baseline_vals) and len(acsmote_vals) > 0:
                    # Calculate differences
                    diffs = acsmote_vals - baseline_vals
                    wins = int(np.sum(diffs > 0))
                    losses = int(np.sum(diffs < 0))
                    ties = int(np.sum(diffs == 0))
                    
                    if np.all(diffs == 0):
                        stat, p = 0.0, 1.0
                    else:
                        stat, p = wilcoxon(acsmote_vals, baseline_vals)
                    
                    stats_results["wilcoxon_acsmote_vs_baseline"][clf][baseline][metric] = {
                        "stat": float(stat) if not np.isnan(stat) else None,
                        "p": float(p) if not np.isnan(p) else None,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties
                    }

    with open('results/statistical_tests.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    print("Saved results/statistical_tests.json")
    
if __name__ == '__main__':
    run_statistical_tests()
