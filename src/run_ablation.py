import os
import csv
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data_loader import load_dataset
from src.methods.ac_smote import ACSMOTE
from src.methods.baselines import get_classifier
from src.utils import compute_all_metrics, set_all_seeds

def run_ablation():
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/ablation_results.csv'
    
    # 5 representative datasets
    ablation_datasets = ['ecoli', 'car_eval_34', 'abalone_19', 'creditcard', 'telco_churn']
    
    # Ablation grid per Section 6.4
    beta_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    k_values = [3, 5, 7]
    
    # Defaults
    default_beta = 0.7
    default_alpha = 2.0
    default_k = 5
    
    # Generate all runs (change one parameter at a time)
    runs = []
    for b in beta_values:
        runs.append({'beta': b, 'alpha': default_alpha, 'k': default_k, 'sweep_param': 'beta'})
    for a in alpha_values:
        if a == default_alpha: continue
        runs.append({'beta': default_beta, 'alpha': a, 'k': default_k, 'sweep_param': 'alpha'})
    for k in k_values:
        if k == default_k: continue
        runs.append({'beta': default_beta, 'alpha': default_alpha, 'k': k, 'sweep_param': 'k'})
        
    random_state = 42
    cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    
    fieldnames = [
        'dataset', 'sweep_param', 'beta', 'alpha', 'k', 'fold_id', 
        'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc'
    ]
    
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for ds_name in ablation_datasets:
            print(f"\nRunning ablation on {ds_name}...")
            X, y = load_dataset(ds_name)
            
            for params in tqdm(runs, desc=f"Ablation sweeps for {ds_name}"):
                
                # Check completed to avoid rerunning if restarted
                # We won't implement full checkpoint reading here to save time, but will just run it
                set_all_seeds(random_state)
                
                for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    
                    resampler = ACSMOTE(
                        alpha=params['alpha'],
                        beta=params['beta'],
                        k_neighbors=params['k'],
                        random_state=random_state
                    )
                    
                    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                    
                    # XGBoost only
                    clf = get_classifier('xgb', random_state)
                    clf.fit(X_train_res, y_train_res)
                    
                    y_pred = clf.predict(X_test)
                    y_score = clf.predict_proba(X_test)[:, 1]
                    
                    metrics = compute_all_metrics(y_test, y_pred, y_score)
                    metrics.update({
                        'dataset': ds_name,
                        'sweep_param': params['sweep_param'],
                        'beta': params['beta'],
                        'alpha': params['alpha'],
                        'k': params['k'],
                        'fold_id': fold_id
                    })
                    
                    writer.writerow(metrics)
                f.flush()

if __name__ == '__main__':
    run_ablation()
