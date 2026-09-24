import pandas as pd
from src.data_loader import load_dataset
datasets = ['ecoli', 'satimage', 'spectrometer', 'car_eval_34', 'oil', 
            'wine_quality', 'mammography', 'abalone_19', 'creditcard', 'telco_churn']
rows = []
for ds in datasets:
    X, y = load_dataset(ds)
    n = len(y)
    pos = sum(y==1)
    neg = sum(y==0)
    ir = neg / pos if pos > 0 else 0
    rows.append(f"{ds} & {n} & {X.shape[1]} & {pos} & {neg} & {ir:.2f} \\\\")
print("\n".join(rows))
