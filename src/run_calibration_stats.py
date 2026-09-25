import os
import pandas as pd
from scipy.stats import wilcoxon

def run_calibration_stats():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/calibration_results.csv')
    
    metrics = ['brier_score', 'ece']
    classifiers = ['rf', 'xgb', 'lr']
    resamplers = ['none', 'smote', 'bsmote', 'adasyn', 'svmsmote', 'smotetomek', 'acsmote']
    baselines = [r for r in resamplers if r != 'acsmote']
    
    df_mean = df.groupby(['dataset', 'classifier', 'resampler'])[metrics].mean().reset_index()
    
    output_lines = []
    output_lines.append("Calibration Metrics (Brier Score and Expected Calibration Error - ECE)\n")
    output_lines.append("Note: For calibration metrics, LOWER is better.\n")
    output_lines.append("Alternative Hypothesis: AC-SMOTE's error is strictly less than the baseline's error.\n")
    
    for clf in classifiers:
        output_lines.append(f"\n{'='*40}")
        output_lines.append(f"CLASSIFIER: {clf.upper()}")
        output_lines.append(f"{'='*40}\n")
        
        clf_data = df_mean[df_mean['classifier'] == clf]
        ac_data = clf_data[clf_data['resampler'] == 'acsmote'].sort_values('dataset')
        
        for base in baselines:
            base_data = clf_data[clf_data['resampler'] == base].sort_values('dataset')
            
            if len(ac_data) == len(base_data) and len(ac_data) > 0:
                output_lines.append(f"--- AC-SMOTE vs {base.upper()} ---")
                for m in metrics:
                    ac_vals = ac_data[m].values
                    base_vals = base_data[m].values
                    diffs = ac_vals - base_vals
                    
                    if all(d == 0 for d in diffs):
                        p = 1.0
                    else:
                        stat, p = wilcoxon(ac_vals, base_vals, alternative='less')
                    
                    output_lines.append(f"Metric: {m.ljust(12)} | AC-SMOTE Mean: {ac_vals.mean():.4f} | {base} Mean: {base_vals.mean():.4f} | p-value (less): {p:.4f}")
                output_lines.append("")

    with open('results/calibration_stats.txt', 'w') as f:
        f.write('\n'.join(output_lines))
        
    print("Saved results/calibration_stats.txt")

if __name__ == '__main__':
    run_calibration_stats()
