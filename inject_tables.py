import re
import pandas as pd
from src.data_loader import load_dataset

# 1. Build Table 1
datasets = ['ecoli', 'satimage', 'spectrometer', 'car_eval_34', 'oil', 
            'wine_quality', 'mammography', 'abalone_19', 'creditcard', 'telco_churn']
t1_rows = []
for ds in datasets:
    X, y = load_dataset(ds)
    n = len(y)
    pos = sum(y==1)
    neg = sum(y==0)
    ir = neg / pos if pos > 0 else 0
    t1_rows.append(f"{ds} & {n} & {X.shape[1]} & {pos} & {neg} & {ir:.2f} \\\\")
table1_content = "\n".join(t1_rows)

table1_latex = f"""
\\begin{{table}}[htbp]
\\caption{{Dataset Characteristics}}
\\label{{tab:datasets}}
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Size}} & \\textbf{{Features}} & \\textbf{{Minority}} & \\textbf{{Majority}} & \\textbf{{IR}} \\\\
\\midrule
{table1_content}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

# 2. Build Table 2
df = pd.read_csv('results/summary_table.csv')
t2_rows = []

def clean_val(v):
    v = str(v)
    parts = re.split(r'[^0-9\.]+', v)
    if len(parts) >= 2:
        return f"{parts[0]} \\pm {parts[1]}"
    return v

for ds in datasets[:5]: # Top 5 to fit page
    for clf in ['rf', 'xgb']:
        sub = df[(df['dataset']==ds) & (df['classifier']==clf)]
        for _, row in sub.iterrows():
            res = row['resampler']
            gm = clean_val(row['gmean'])
            f1 = clean_val(row['f1'])
            bal = clean_val(row['bal_acc'])
            t2_rows.append(f"{ds} & {clf} & {res} & ${gm}$ & ${f1}$ & ${bal}$ \\\\")
            
table2_content = "\n".join(t2_rows)
table2_latex = f"""
\\begin{{table*}}[htbp]
\\caption{{Main Results (Sample of 5 Datasets, RF and XGBoost) reported as Mean $\\pm$ Std.}}
\\label{{tab:results}}
\\centering
\\begin{{tabular}}{{lllrrr}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Classifier}} & \\textbf{{Resampler}} & \\textbf{{G-mean}} & \\textbf{{F1-score}} & \\textbf{{Bal. Acc.}} \\\\
\\midrule
{table2_content}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

with open('paper/main.tex', 'r') as f:
    content = f.read()

# Insert tables into Section 4 (Experimental Setup) and Section 5 (Results)
content = content.replace('\\subsection{Datasets and Baselines}', '\\subsection{Datasets and Baselines}\n' + table1_latex + '\nAs summarized in Table \\ref{tab:datasets}, we benchmark AC-SMOTE...')
content = content.replace('folds. \n\n\\subsection{Performance across Classifiers}', 'folds. Detailed metrics are provided in Table \\ref{tab:results}.\n\n' + table2_latex + '\n\\subsection{Performance across Classifiers}')

with open('paper/main.tex', 'w') as f:
    f.write(content)
print("Tables injected.")
