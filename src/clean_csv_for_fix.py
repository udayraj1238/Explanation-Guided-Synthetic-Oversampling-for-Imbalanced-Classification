import pandas as pd
import os

datasets = ['satimage', 'wine_quality', 'mammography', 'abalone_19', 'creditcard', 'telco_churn']
methods = ['svmsmote', 'smotetomek']

def clean_csv(file_path):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    mask = (df['dataset'].isin(datasets)) & (df['resampler'].isin(methods))
    dropped = mask.sum()
    if dropped > 0:
        df = df[~mask]
        df.to_csv(file_path, index=False)
        print(f"Dropped {dropped} rows from {file_path}")
    else:
        print(f"No rows needed dropping in {file_path}")

clean_csv('results/raw_results.csv')
clean_csv('results/lr_scaled_results.csv')
clean_csv('results/boundary_fidelity_results.csv')
clean_csv('results/calibration_results.csv')
