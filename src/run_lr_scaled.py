import os
import csv
import time
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_all_datasets
from src.utils import set_all_seeds, compute_all_metrics
from src.methods.baselines import get_resampler

def run_lr_scaled():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/lr_scaled_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We only run logreg, so classifier is fixed
                key = (row['dataset'], row['resampler'], int(row['fold_id']))
                completed.add(key)
                
    datasets = load_all_datasets()
    dataset_names = list(datasets.keys())
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote', 'svmsmote', 'smotetomek']
    seed = 42
    
    grid = list(product(dataset_names, resamplers))
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'resampler', 'fold_id', 
                  'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds', 'resampler_failed']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print(f"Total grid combinations for LR-Scaled: {len(grid)}")
        
        for ds_name, res_name in tqdm(grid, desc="LR-Scaled Experiments"):
            done_folds = {fold_id for (d, r, fold_id) in completed if d == ds_name and r == res_name}
            
            if len(done_folds) == 15:
                continue
                
            X, y = datasets[ds_name]
            set_all_seeds(seed)
            cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
            
            for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                if fold_id in done_folds:
                    continue
                    
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                # 1. Leakage-safe scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 2. Resample
                if res_name != 'none':
                    resampler = get_resampler(res_name, seed)
                    start_time = time.time()
                    try:
                        X_train_res, y_train_res = resampler.fit_resample(X_train_scaled, y_train)
                    except Exception as e:
                        print(f"Warning: {res_name} failed on {ds_name} fold {fold_id} (error: {e})")
                        X_train_res, y_train_res = X_train_scaled, y_train
                    fit_time = time.time() - start_time
                else:
                    start_time = time.time()
                    X_train_res, y_train_res = X_train_scaled, y_train
                    fit_time = time.time() - start_time
                    
                # 3. Fit LR
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train_res, y_train_res)
                
                # 4. Evaluate
                y_pred = clf.predict(X_test_scaled)
                y_score = clf.predict_proba(X_test_scaled)[:, 1]
                
                metrics = compute_all_metrics(y_test, y_pred, y_score)
                metrics.update({
                    'dataset': ds_name,
                    'classifier': 'logreg_scaled',
                    'resampler': res_name,
                    'fold_id': fold_id,
                    'fit_time_seconds': fit_time
                })
                
                writer.writerow(metrics)
                f.flush()
                completed.add((ds_name, res_name, fold_id))

if __name__ == '__main__':
    run_lr_scaled()
