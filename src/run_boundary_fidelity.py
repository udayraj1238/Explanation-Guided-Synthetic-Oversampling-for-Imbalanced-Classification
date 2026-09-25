import os
import csv
import numpy as np
import scipy.stats
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_all_datasets
from src.utils import set_all_seeds, get_shap_importance
from src.methods.baselines import get_resampler

def get_classifier(name, seed):
    if name == 'rf':
        # Added max_depth=10 because SHAP exact calculation on fully grown RFs takes hours.
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    raise ValueError("Unknown classifier")

def run_boundary_fidelity():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/boundary_fidelity_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], row['resampler'], int(row['fold_id']))
                completed.add(key)
                
    datasets = load_all_datasets()
    dataset_names = list(datasets.keys())
    classifiers = ['rf', 'xgb']
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'svmsmote', 'smotetomek', 'acsmote']
    seed = 42
    
    grid = list(product(dataset_names, classifiers))
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'resampler', 'fold_id', 'spearman_rho', 'spearman_p']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print("Starting boundary fidelity experiments...")
        
        for ds_name, clf_name in tqdm(grid, desc="Boundary Fidelity Grid"):
            X, y = datasets[ds_name]
            set_all_seeds(seed)
            cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
            
            for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                fold_resamplers_done = {r for (d, c, r, fold) in completed if d == ds_name and c == clf_name and fold == fold_id}
                if len(fold_resamplers_done) == len(resamplers):
                    continue
                    
                X_train, y_train = X[train_idx], y[train_idx]
                X_test = X[test_idx]
                
                if len(X_test) > 500:
                    idx = np.random.RandomState(seed).choice(len(X_test), 500, replace=False)
                    X_test_shap = X_test[idx]
                else:
                    X_test_shap = X_test
                    
                clf_ref = get_classifier(clf_name, seed).fit(X_train, y_train)
                ref_importance = get_shap_importance(clf_ref, X_test_shap)
                
                for res_name in resamplers:
                    if res_name in fold_resamplers_done:
                        continue
                        
                    if res_name != 'none':
                        resampler = get_resampler(res_name, seed)
                        try:
                            X_res, y_res = resampler.fit_resample(X_train, y_train)
                        except Exception as e:
                            print(f"Warning: {res_name} failed on {ds_name} fold {fold_id} (error: {e})")
                            X_res, y_res = X_train, y_train
                    else:
                        X_res, y_res = X_train, y_train
                        
                    clf_M = get_classifier(clf_name, seed).fit(X_res, y_res)
                    M_importance = get_shap_importance(clf_M, X_test_shap)
                    
                    rho, p = scipy.stats.spearmanr(ref_importance, M_importance)
                    
                    # Handle edge cases where features have zero variance or all identical SHAP values
                    if np.isnan(rho):
                        rho = 0.0
                        p = 1.0
                        
                    writer.writerow({
                        'dataset': ds_name,
                        'classifier': clf_name,
                        'resampler': res_name,
                        'fold_id': fold_id,
                        'spearman_rho': float(rho),
                        'spearman_p': float(p)
                    })
                    f.flush()
                    completed.add((ds_name, clf_name, res_name, fold_id))

if __name__ == '__main__':
    run_boundary_fidelity()
