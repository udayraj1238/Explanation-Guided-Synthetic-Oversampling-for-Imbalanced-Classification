import os
import csv
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data_loader import load_all_datasets
from src.experiment import run_single_cell
from src.utils import set_all_seeds

def main(dry_run=False):
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/raw_results.csv'
    
    # 1. Load checkpoints
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['classifier'], row['resampler'], int(row['fold_id']))
                completed.add(key)
                
    # 2. Setup grid
    datasets = load_all_datasets()
    classifiers = ['logreg', 'rf', 'xgb']
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
    
    if dry_run:
        # 2 datasets, 1 classifier, 6 resamplers
        dataset_names = list(datasets.keys())[:2]
        classifiers = ['logreg']
    else:
        dataset_names = list(datasets.keys())
        
    cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    random_state = 42
    
    grid = list(product(dataset_names, classifiers, resamplers))
    
    # Check if CSV needs header
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    
    fieldnames = [
        'dataset', 'classifier', 'resampler', 'fold_id', 
        'f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc', 'fit_time_seconds'
    ]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        print(f"Total grid combinations: {len(grid)}")
        
        # We loop over combinations
        for ds_name, clf_name, res_name in tqdm(grid, desc="Experiments"):
            # Check which folds are already done for this combination
            done_folds_for_combo = {fold_id for (d, c, r, fold_id) in completed 
                                    if d == ds_name and c == clf_name and r == res_name}
            
            if len(done_folds_for_combo) == 15:
                continue
                
            X, y = datasets[ds_name]
            set_all_seeds(random_state)
            
            try:
                results = run_single_cell(
                    X, y, ds_name, clf_name, res_name, 
                    cv_splitter, random_state, 
                    completed_folds=done_folds_for_combo
                )
                
                # Append to CSV immediately
                for r in results:
                    writer.writerow(r)
                f.flush()
            except Exception as e:
                print(f"\nError in combination {ds_name} {clf_name} {res_name}: {e}")

if __name__ == '__main__':
    import sys
    dry = '--dry-run' in sys.argv
    main(dry_run=dry)
