import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.metrics import geometric_mean_score

from src.data_loader import load_all_datasets
from src.utils import set_all_seeds, compute_all_metrics
from src.methods.ac_smote import ACSMOTE

def get_classifier(name, seed):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    raise ValueError("Unknown classifier")

def run_acsmote_tuning():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/acsmote_tuned_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], int(row['fold_id']))
                completed.add(key)
                
    datasets = load_all_datasets()
    dataset_names = list(datasets.keys())
    classifiers = ['rf', 'xgb']
    
    alpha_grid = [1.0, 2.0, 4.0]
    beta_grid = [0.3, 0.7, 1.0]
    param_grid = list(product(alpha_grid, beta_grid))
    
    outer_seed = 42
    
    grid = list(product(dataset_names, classifiers))
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'fold_id', 'alpha_star', 'beta_star', 
                  'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'resampler_failed']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print("Starting ACSMOTE Nested Tuning CV...")
        
        for ds_name, clf_name in tqdm(grid, desc="Tuning ACSMOTE"):
            X, y = datasets[ds_name]
            set_all_seeds(outer_seed)
            cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=outer_seed)
            
            for fold_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
                if (ds_name, clf_name, fold_id) in completed:
                    continue
                    
                X_train_outer, y_train_outer = X[train_idx], y[train_idx]
                X_test_outer, y_test_outer = X[test_idx], y[test_idx]
                
                # 1. Inner Split (leakage-safe)
                inner_seed = outer_seed + fold_id
                
                X_inner_train_full, X_inner_val, y_inner_train_full, y_inner_val = train_test_split(
                    X_train_outer, y_train_outer, test_size=0.25, 
                    stratify=y_train_outer, random_state=inner_seed
                )
                
                # Safe Stratified Subsampling specifically for the nested inner loop to cap tuning runtime.
                # Never touches X_train_outer or X_test_outer!
                if len(X_inner_train_full) > 15000:
                    _, X_inner_train, _, y_inner_train = train_test_split(
                        X_inner_train_full, y_inner_train_full, test_size=15000, 
                        stratify=y_inner_train_full, random_state=inner_seed
                    )
                else:
                    X_inner_train = X_inner_train_full
                    y_inner_train = y_inner_train_full
                
                best_gmean = -1.0
                best_alpha = None
                best_beta = None
                
                # 2. Grid Search on Inner Split
                for alpha, beta in param_grid:
                    try:
                        resampler = ACSMOTE(alpha=alpha, beta=beta, random_state=inner_seed)
                        X_res_inner, y_res_inner = resampler.fit_resample(X_inner_train, y_inner_train)
                        
                        clf_inner = get_classifier(clf_name, inner_seed).fit(X_res_inner, y_res_inner)
                        y_pred_val = clf_inner.predict(X_inner_val)
                        
                        gmean_val = float(geometric_mean_score(y_inner_val, y_pred_val))
                        
                        if gmean_val > best_gmean:
                            best_gmean = gmean_val
                            best_alpha = alpha
                            best_beta = beta
                    except Exception as e:
                        print(f"Warning: tuning failed for {ds_name} alpha={alpha} beta={beta}: {e}")
                
                # Fallback to default if all failed (extreme minority classes might prevent resampling entirely)
                if best_alpha is None:
                    best_alpha, best_beta = 2.0, 0.7
                    
                # 4. Refit ACSMOTE on FULL X_train_outer with optimal parameters
                final_resampler = ACSMOTE(alpha=best_alpha, beta=best_beta, random_state=outer_seed)
                try:
                    X_res_outer, y_res_outer = final_resampler.fit_resample(X_train_outer, y_train_outer)
                except Exception as e:
                    X_res_outer, y_res_outer = X_train_outer, y_train_outer
                    
                clf_final = get_classifier(clf_name, outer_seed).fit(X_res_outer, y_res_outer)
                
                # Predict on X_test_outer
                y_pred = clf_final.predict(X_test_outer)
                if hasattr(clf_final, "predict_proba"):
                    y_score = clf_final.predict_proba(X_test_outer)[:, 1]
                else:
                    y_score = clf_final.decision_function(X_test_outer)
                    
                metrics = compute_all_metrics(y_test_outer, y_pred, y_score)
                
                # 5. Record final validated result
                row = {
                    'dataset': ds_name,
                    'classifier': clf_name,
                    'fold_id': fold_id,
                    'alpha_star': best_alpha,
                    'beta_star': best_beta,
                    'f1': metrics['f1'],
                    'auc_pr': metrics['auc_pr'],
                    'mcc': metrics['mcc'],
                    'gmean': metrics['gmean'],
                    'bal_acc': metrics['bal_acc']
                }
                
                writer.writerow(row)
                f.flush()
                completed.add((ds_name, clf_name, fold_id))

if __name__ == '__main__':
    run_acsmote_tuning()
