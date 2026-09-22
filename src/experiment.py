import time
import numpy as np
from src.methods.baselines import get_resampler, get_classifier
from src.utils import compute_all_metrics

def run_single_cell(X, y, dataset_name, classifier_name, resampler_name,
                    cv_splitter, random_state, completed_folds=None) -> list[dict]:
    """
    Runs the full CV loop for one (dataset, classifier, resampler) combination.
    For each of the 15 splits: split first, fit resampler on train fold only
    (skip this step entirely if resampler_name == 'none'), fit classifier on
    the resampled train fold, evaluate on the untouched original test fold.
    Returns one dict per fold matching the raw_results.csv schema (Section 8),
    each also tagged with dataset/classifier/resampler/fold_id.
    """
    if completed_folds is None:
        completed_folds = set()
        
    results = []
    
    for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        if fold_id in completed_folds:
            continue
            
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        start_time = time.time()
        
        # 1. Resample (ONLY ON TRAIN)
        if resampler_name != 'none':
            resampler = get_resampler(resampler_name, random_state)
            X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
            
        # 2. Fit Classifier (ON RESAMPLED TRAIN)
        clf = get_classifier(classifier_name, random_state)
        clf.fit(X_train_res, y_train_res)
        
        # 3. Evaluate (ON ORIGINAL UNTOUCHED TEST)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
            
        fit_time = time.time() - start_time
        
        # 4. Metrics
        metrics = compute_all_metrics(y_test, y_pred, y_score)
        
        # Tag the row
        metrics.update({
            "dataset": dataset_name,
            "classifier": classifier_name,
            "resampler": resampler_name,
            "fold_id": fold_id,
            "fit_time_seconds": fit_time
        })
        
        results.append(metrics)
        
    return results
