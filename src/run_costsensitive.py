import os
import csv
import time
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_all_datasets
from src.utils import set_all_seeds, compute_all_metrics

def get_cost_sensitive_classifier(name, seed, y_train):
    if name == 'logreg':
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    elif name == 'rf':
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, class_weight='balanced')
    elif name == 'xgb':
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
        return XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', 
                             n_jobs=-1, random_state=seed, scale_pos_weight=scale_pos)
    else:
        raise ValueError(f"Unknown classifier: {name}")

def run_costsensitive():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/costsensitive_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], int(row['fold_id']))
                completed.add(key)
                
    datasets = load_all_datasets()
    dataset_names = list(datasets.keys())
    classifiers = ['logreg', 'rf', 'xgb']
    seed = 42
    
    grid = list(product(dataset_names, classifiers))
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'resampler', 'fold_id', 
                  'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print(f"Total grid combinations for cost-sensitive: {len(grid)}")
        
        for ds_name, clf_name in tqdm(grid, desc="Cost-sensitive Experiments"):
            done_folds = {fold_id for (d, c, fold_id) in completed 
                          if d == ds_name and c == clf_name}
            
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
                
                # Fit
                start_time = time.time()
                clf = get_cost_sensitive_classifier(clf_name, seed, y_train)
                clf.fit(X_train, y_train)
                fit_time = time.time() - start_time
                
                # Evaluate
                y_pred = clf.predict(X_test)
                y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)
                
                metrics = compute_all_metrics(y_test, y_pred, y_score)
                metrics.update({
                    'dataset': ds_name,
                    'classifier': clf_name,
                    'resampler': 'cost_sensitive',
                    'fold_id': fold_id,
                    'fit_time_seconds': fit_time
                })
                
                writer.writerow(metrics)
                f.flush()
                completed.add((ds_name, clf_name, fold_id))

if __name__ == '__main__':
    run_costsensitive()
