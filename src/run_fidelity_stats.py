import os
import pandas as pd
from scipy.stats import wilcoxon

def run_fidelity_stats():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/boundary_fidelity_results.csv')
    
    # We want the dataset-level mean Spearman rho across the 15 folds
    df_mean = df.groupby(['dataset', 'classifier', 'resampler'])['spearman_rho'].mean().reset_index()
    
    classifiers = df['classifier'].unique()
    baselines = [r for r in df['resampler'].unique() if r not in ['none', 'acsmote']]
    
    output_lines = []
    output_lines.append("Decision Boundary Fidelity (Spearman Rho) - Statistical Tests\n")
    output_lines.append("Null Hypothesis: AC-SMOTE's Spearman rho is <= the baseline's rho")
    output_lines.append("Alternative Hypothesis: AC-SMOTE's Spearman rho is > the baseline's rho (better fidelity)\n")
    
    from statsmodels.stats.multitest import multipletests
    
    for clf in classifiers:
        output_lines.append(f"\n{'='*40}")
        output_lines.append(f"CLASSIFIER: {clf.upper()}")
        output_lines.append(f"{'='*40}\n")
        
        ac_data = df_mean[(df_mean['classifier'] == clf) & (df_mean['resampler'] == 'acsmote')].sort_values('dataset')
        
        none_data = df_mean[(df_mean['classifier'] == clf) & (df_mean['resampler'] == 'none')]
        output_lines.append(f"Reference upper-bound (none): mean rho = {none_data['spearman_rho'].mean():.4f}\n")
        
        p_vals = []
        comparisons = []
        
        for base in baselines:
            base_data = df_mean[(df_mean['classifier'] == clf) & (df_mean['resampler'] == base)].sort_values('dataset')
            
            if len(ac_data) == len(base_data) and len(ac_data) > 0:
                diffs = ac_data['spearman_rho'].values - base_data['spearman_rho'].values
                wins = sum(diffs > 0)
                losses = sum(diffs < 0)
                ties = sum(diffs == 0)
                
                if all(d == 0 for d in diffs):
                    p = 1.0
                else:
                    stat, p = wilcoxon(ac_data['spearman_rho'].values, base_data['spearman_rho'].values, alternative='greater')
                
                ac_mean = ac_data['spearman_rho'].mean()
                base_mean = base_data['spearman_rho'].mean()
                
                comparisons.append({
                    'base': base,
                    'ac_mean': ac_mean,
                    'base_mean': base_mean,
                    'wins': wins,
                    'losses': losses,
                    'ties': ties,
                    'p': p
                })
                p_vals.append(p)
                
        if len(p_vals) > 0:
            _, p_holm, _, _ = multipletests(p_vals, method='holm')
            _, p_bh, _, _ = multipletests(p_vals, method='fdr_bh')
            
            for i, comp in enumerate(comparisons):
                result_str = (
                    f"AC-SMOTE vs {comp['base'].ljust(12)} | "
                    f"Mean Rho: {comp['ac_mean']:.4f} vs {comp['base_mean']:.4f} | "
                    f"W:{comp['wins']:2d} L:{comp['losses']:2d} T:{comp['ties']:2d} | "
                    f"p_raw: {comp['p']:.4f} | p_holm: {p_holm[i]:.4f} | p_fdr: {p_bh[i]:.4f}"
                )
                output_lines.append(result_str)
                
    with open('results/boundary_fidelity_stats.txt', 'w') as f:
        f.write('\n'.join(output_lines))
        
    print("Saved results/boundary_fidelity_stats.txt")

if __name__ == '__main__':
    run_fidelity_stats()
