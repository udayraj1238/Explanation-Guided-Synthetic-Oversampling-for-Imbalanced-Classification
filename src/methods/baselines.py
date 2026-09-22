from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_resampler(name: str, random_state: int):
    """
    name in {'none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote'}
    Returns an object with .fit_resample(X, y) -> (X_res, y_res),
    or None for 'none' (meaning: use the raw imbalanced training fold as-is).
    """
    if name == 'none':
        return None
    elif name == 'ros':
        return RandomOverSampler(random_state=random_state)
    elif name == 'smote':
        return SMOTE(random_state=random_state)
    elif name == 'bsmote':
        return BorderlineSMOTE(random_state=random_state)
    elif name == 'adasyn':
        return ADASYN(random_state=random_state)
    elif name == 'acsmote':
        # Will be implemented fully on Day 2
        try:
            from src.methods.ac_smote import ACSMOTE
            return ACSMOTE(random_state=random_state)
        except ImportError:
            raise NotImplementedError("ACSMOTE is not yet implemented.")
    else:
        raise ValueError(f"Unknown resampler: {name}")

def get_classifier(name: str, random_state: int):
    if name == 'logreg':
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif name == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {name}")
