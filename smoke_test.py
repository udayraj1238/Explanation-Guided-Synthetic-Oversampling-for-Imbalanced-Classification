import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.utils import compute_all_metrics, set_all_seeds
from src.methods.baselines import get_resampler, get_classifier

def run_smoke_test():
    set_all_seeds(42)
    dataset_name = 'ecoli'
    print(f"Loading {dataset_name}...")
    X, y = load_dataset(dataset_name)
    print(f"Dataset shape: {X.shape}, minority ratio: {np.sum(y)}/{len(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    classifier_name = 'logreg'
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn']
    
    for resampler_name in resamplers:
        print(f"\n--- Testing Resampler: {resampler_name} ---")
        resampler = get_resampler(resampler_name, random_state=42)
        
        if resampler is not None:
            X_res, y_res = resampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
            
        print(f"Resampled train shape: {X_res.shape}, minority ratio: {np.sum(y_res)}/{len(y_res)}")
        
        clf = get_classifier(classifier_name, random_state=42)
        clf.fit(X_res, y_res)
        
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
            
        metrics = compute_all_metrics(y_test, y_pred, y_score)
        print(f"Metrics: {metrics}")

if __name__ == '__main__':
    run_smoke_test()
