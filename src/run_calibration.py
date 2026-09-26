import os
import csv
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_dataset
from src.utils import set_all_seeds
from src.methods.baselines import get_resampler

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard ECE: bin predicted probabilities into n_bins equal-width bins,
    for each bin compute |mean_predicted_prob - actual_fraction_positive|,
    weight by the bin's share of samples, sum across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1] if i < n_bins-1 else y_prob <= bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

def get_classifier(name, seed):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    elif name == 'lr':
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError("Unknown classifier")

def run_calibration():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/calibration_results.csv'
    
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], row['resampler'], int(row['fold_id']))
                completed.add(key)
                
    datasets = ['ecoli', 'car_eval_34', 'abalone_19', 'creditcard', 'telco_churn']
    classifiers = ['rf', 'xgb', 'lr']
    resamplers = ['none', 'smote', 'bsmote', 'adasyn', 'svmsmote', 'smotetomek', 'acsmote']
    seed = 42
    
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    fieldnames = ['dataset', 'classifier', 'resampler', 'fold_id', 'brier_score', 'ece']
                  
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for ds_name in datasets:
            X_full, y_full = load_dataset(ds_name)
            set_all_seeds(seed)
            cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
            
            ds_grid = [(c, r) for c in classifiers for r in resamplers]
            
            for clf_name, res_name in tqdm(ds_grid, desc=f"Calibration Grid ({ds_name})"):
                for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X_full, y_full)):
                    if (ds_name, clf_name, res_name, fold_id) in completed:
                        continue
                        
                    X_train, y_train = X_full[train_idx], y_full[train_idx]
                    X_test, y_test = X_full[test_idx], y_full[test_idx]
                    
                    if clf_name == 'lr':
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                    if res_name != 'none':
                        resampler = get_resampler(res_name, seed)
                        try:
                            X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                        except Exception as e:
                            print(f"Warning: {res_name} failed on {ds_name} fold {fold_id} (error: {e})")
                            X_train_res, y_train_res = X_train, y_train
                    else:
                        X_train_res, y_train_res = X_train, y_train
                        
                    clf = get_classifier(clf_name, seed)
                    clf.fit(X_train_res, y_train_res)
                    
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    
                    brier = brier_score_loss(y_test, y_prob)
                    ece = expected_calibration_error(y_test, y_prob)
                    
                    writer.writerow({
                        'dataset': ds_name,
                        'classifier': clf_name,
                        'resampler': res_name,
                        'fold_id': fold_id,
                        'brier_score': float(brier),
                        'ece': float(ece)
                    })
                    f.flush()
                    completed.add((ds_name, clf_name, res_name, fold_id))

if __name__ == '__main__':
    run_calibration()
