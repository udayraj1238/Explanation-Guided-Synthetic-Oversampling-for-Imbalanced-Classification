import pandas as pd
from scipy.stats import wilcoxon
import os

def compare_tuned():
    os.makedirs('results', exist_ok=True)
    tuned = pd.read_csv('results/acsmote_tuned_results.csv')
    raw = pd.read_csv('results/raw_results.csv')
    
    tuned_mean = tuned.groupby(['dataset', 'classifier'])[['f1', 'mcc', 'gmean', 'bal_acc']].mean().reset_index()
    tuned_mean['resampler'] = 'acsmote_tuned'
    
    raw_mean = raw.groupby(['dataset', 'classifier', 'resampler'])[['f1', 'mcc', 'gmean', 'bal_acc']].mean().reset_index()
    untuned_acsmote = raw_mean[raw_mean['resampler'] == 'acsmote']
    smotetomek = raw_mean[raw_mean['resampler'] == 'smotetomek']
    
    metrics = ['gmean', 'f1', 'mcc']
    classifiers = ['rf', 'xgb']
    
    output_lines = []
    
    output_lines.append("=== TUNED AC-SMOTE vs UNTUNED AC-SMOTE ===")
    for clf in classifiers:
        output_lines.append(f"\nClassifier: {clf.upper()}")
        t_data = tuned_mean[tuned_mean['classifier'] == clf].sort_values('dataset')
        u_data = untuned_acsmote[untuned_acsmote['classifier'] == clf].sort_values('dataset')
        
        for m in metrics:
            # check for zero variance (identical means)
            diffs = t_data[m].values - u_data[m].values
            if all(d == 0 for d in diffs):
                p = 1.0
            else:
                stat, p = wilcoxon(t_data[m].values, u_data[m].values, alternative='greater')
            output_lines.append(f"Metric: {m.ljust(5)} | Tuned Mean: {t_data[m].mean():.4f} | Untuned Mean: {u_data[m].mean():.4f} | p-val (greater): {p:.4f}")
            
    output_lines.append("\n=== TUNED AC-SMOTE vs SMOTETOMEK ===")
    for clf in classifiers:
        output_lines.append(f"\nClassifier: {clf.upper()}")
        t_data = tuned_mean[tuned_mean['classifier'] == clf].sort_values('dataset')
        s_data = smotetomek[smotetomek['classifier'] == clf].sort_values('dataset')
        
        for m in metrics:
            diffs = t_data[m].values - s_data[m].values
            if all(d == 0 for d in diffs):
                p = 1.0
            else:
                stat, p = wilcoxon(t_data[m].values, s_data[m].values, alternative='greater')
            output_lines.append(f"Metric: {m.ljust(5)} | Tuned Mean: {t_data[m].mean():.4f} | SMOTETomek Mean: {s_data[m].mean():.4f} | p-val (greater): {p:.4f}")
            
    output_lines.append("\n=== HYPERPARAMETER CONSISTENCY ===")
    output_lines.append("Most frequently selected (alpha, beta) across the 15 folds:\n")
    mode_params = tuned.groupby(['dataset', 'classifier'])[['alpha_star', 'beta_star']].agg(lambda x: x.mode().iloc[0])
    output_lines.append(mode_params.to_string())
    
    with open('results/acsmote_tuning_stats.txt', 'w') as f:
        f.write('\n'.join(output_lines))
        
    print("Saved tuning analysis to results/acsmote_tuning_stats.txt")

if __name__ == '__main__':
    compare_tuned()
