import os
import pandas as pd
from scipy.stats import rankdata

def generate_table2():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('results/raw_results.csv')
    
    # Calculate dataset-level means across the 15 folds
    metrics = ['f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc']
    group_cols = ['dataset', 'classifier', 'resampler']
    
    summary_mean = df.groupby(group_cols)[metrics].mean().reset_index()
    
    # We want to rank the resamplers for each (dataset, classifier, metric).
    # Higher metric is better, so we rank ascending=False
    for m in metrics:
        summary_mean[f'{m}_rank'] = summary_mean.groupby(['dataset', 'classifier'])[m].rank(ascending=False)
        
    ranked_df = summary_mean
    
    # Now average the ranks across all 10 datasets for each classifier and resampler
    mean_ranks = ranked_df.groupby(['classifier', 'resampler'])[[f'{m}_rank' for m in metrics]].mean().reset_index()
    
    # Format into LaTeX table
    classifiers = ['logreg', 'rf', 'xgb']
    clf_names = {'logreg': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'svmsmote', 'smotetomek', 'acsmote']
    res_names = {'none': 'None', 'ros': 'ROS', 'smote': 'SMOTE', 'bsmote': 'B-SMOTE', 
                 'adasyn': 'ADASYN', 'svmsmote': 'SVMSMOTE', 'smotetomek': 'SMOTETomek', 'acsmote': 'AC-SMOTE'}
                 
    latex_out = []
    latex_out.append("\\begin{table*}[htbp]")
    latex_out.append("\\caption{Mean Ranks of Resampling Methods Across All 10 Datasets (Lower is Better)}")
    latex_out.append("\\label{tab:mean_ranks}")
    latex_out.append("\\centering")
    latex_out.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    latex_out.append("\\toprule")
    latex_out.append("Classifier & Method & F1-Score & AUC-PR & MCC & G-Mean & Bal. Acc \\\\")
    latex_out.append("\\midrule")
    
    for clf in classifiers:
        for i, res in enumerate(resamplers):
            clf_str = clf_names[clf] if i == 0 else ""
            row_data = mean_ranks[(mean_ranks['classifier'] == clf) & (mean_ranks['resampler'] == res)]
            if row_data.empty:
                continue
            
            # Format the numbers
            row_str = f"{clf_str} & {res_names[res]}"
            for m in metrics:
                val = row_data[f'{m}_rank'].values[0]
                # highlight best rank in bold (minimum rank)
                min_val = mean_ranks[mean_ranks['classifier'] == clf][f'{m}_rank'].min()
                if abs(val - min_val) < 1e-5:
                    row_str += f" & \\textbf{{{val:.2f}}}"
                else:
                    row_str += f" & {val:.2f}"
            row_str += " \\\\"
            latex_out.append(row_str)
        latex_out.append("\\midrule")
        
    # Remove last midrule and close table
    latex_out.pop()
    latex_out.append("\\bottomrule")
    latex_out.append("\\end{tabular}")
    latex_out.append("\\end{table*}")
    
    with open('results/table2_mean_ranks.tex', 'w') as f:
        f.write('\n'.join(latex_out))
        
    print("Generated results/table2_mean_ranks.tex")

if __name__ == '__main__':
    generate_table2()
