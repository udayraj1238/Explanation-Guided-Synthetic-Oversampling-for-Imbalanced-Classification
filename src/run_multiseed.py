import os
import csv
import time
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data_loader import load_all_datasets
from src.experiment import run_single_cell
from src.utils import set_all_seeds, compute_all_metrics
from src.methods.baselines import get_resampler, get_classifier

def run_multiseed():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/multiseed_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], row['resampler'], int(row['seed']), int(row['fold_id']))
                completed.add(key)
                
    datasets = load_all_datasets()
    dataset_names = list(datasets.keys())
    classifiers = ['logreg', 'rf', 'xgb']
    resamplers = ['none', 'smote', 'bsmote', 'adasyn', 'acsmote']
    seeds = [123, 2026]
    
    grid = list(product(dataset_names, classifiers, resamplers, seeds))
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'resampler', 'seed', 'fold_id', 
                  'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print(f"Total grid combinations for multi-seed: {len(grid)}")
        
        for ds_name, clf_name, res_name, seed in tqdm(grid, desc="Multi-seed Experiments"):
            done_folds = {fold_id for (d, c, r, s, fold_id) in completed 
                          if d == ds_name and c == clf_name and r == res_name and s == seed}
            
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
                
                # Resample
                if res_name != 'none':
                    resampler = get_resampler(res_name, seed)
                    start_time = time.time()
                    try:
                        X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                    except Exception as e:
                        print(f"Warning: {res_name} failed on {ds_name} seed {seed} fold {fold_id} (error: {e})")
                        X_train_res, y_train_res = X_train, y_train
                    fit_time = time.time() - start_time
                else:
                    start_time = time.time()
                    X_train_res, y_train_res = X_train, y_train
                    fit_time = time.time() - start_time
                    
                # Classifier
                clf = get_classifier(clf_name, seed)
                clf.fit(X_train_res, y_train_res)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)
                
                metrics = compute_all_metrics(y_test, y_pred, y_score)
                metrics.update({
                    'dataset': ds_name,
                    'classifier': clf_name,
                    'resampler': res_name,
                    'seed': seed,
                    'fold_id': fold_id,
                    'fit_time_seconds': fit_time
                })
                
                writer.writerow(metrics)
                f.flush()
                completed.add((ds_name, clf_name, res_name, seed, fold_id))

if __name__ == '__main__':
    run_multiseed()
