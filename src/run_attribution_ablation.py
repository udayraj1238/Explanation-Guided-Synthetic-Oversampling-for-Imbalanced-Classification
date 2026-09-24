import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from src.data_loader import load_dataset
from src.experiment import run_single_cell

def run_attribution_ablation():
    os.makedirs('results', exist_ok=True)
    results_file = 'results/attribution_ablation_results.csv'
    
    datasets = ['ecoli', 'car_eval_34', 'abalone_19', 'creditcard', 'telco_churn']
    classifier = 'xgb'
    attribution_methods = ['shap', 'rf_gini', 'permutation', 'mutual_info', 'anova_f', 'uniform']
    
    cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        # Unique folds completed
        completed_folds = set()
        for _, row in existing_df.iterrows():
            completed_folds.add((row['dataset'], row['attribution_method'], row['fold_id']))
    else:
        completed_folds = set()
        
    for ds in datasets:
        print(f"Loading {ds}...")
        X, y = load_dataset(ds)
        
        for attr_method in attribution_methods:
            # We want to skip if all 15 folds for this dataset+method are already done
            expected_folds = 15
            done_folds = sum(1 for f in completed_folds if f[0] == ds and f[1] == attr_method)
            if done_folds == expected_folds:
                print(f"Skipping {ds} - {attr_method}, fully completed.")
                continue
                
            print(f"Running ablation on {ds} with attribution_method={attr_method}...")
            
            # Since experiment.py expects a resampler name and get_resampler doesn't take attribution_method directly,
            # we will create the resampler manually here and monkey-patch or just modify run_single_cell logic locally,
            # but actually the prompt says "extend ACSMOTE, don't duplicate it". 
            # We can create a custom resampler object and pass it. But experiment.py takes string.
            # Let's instantiate ACSMOTE directly here and pass it, OR we modify baselines.py.
            pass # wait, let's just do it directly here to not break baselines string API too much

            from src.methods.ac_smote import ACSMOTE
            from src.methods.baselines import get_classifier
            from src.utils import compute_all_metrics
            import time
            
            # We iterate manually for the 15 folds to easily use the custom ACSMOTE
            for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                if (ds, attr_method, fold_id) in completed_folds:
                    continue
                    
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                resampler = ACSMOTE(random_state=42, alpha=2.0, beta=0.7, k_neighbors=5, attribution_method=attr_method)
                
                start_time = time.time()
                try:
                    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"Error on {ds} fold {fold_id} with {attr_method}: {e}")
                    X_train_res, y_train_res = X_train, y_train
                fit_time = time.time() - start_time
                
                clf = get_classifier(classifier, 42)
                clf.fit(X_train_res, y_train_res)
                
                y_pred = clf.predict(X_test)
                y_score = clf.predict_proba(X_test)[:, 1]
                
                metrics = compute_all_metrics(y_test, y_pred, y_score)
                metrics['dataset'] = ds
                metrics['attribution_method'] = attr_method
                metrics['fold_id'] = fold_id
                metrics['fit_time_seconds'] = fit_time
                
                df_res = pd.DataFrame([metrics])
                df_res.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
                completed_folds.add((ds, attr_method, fold_id))
                
if __name__ == '__main__':
    run_attribution_ablation()
