from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class SafeSVMSMOTE:
    def __init__(self, random_state):
        self.random_state = random_state
        self.svmsmote = SVMSMOTE(random_state=random_state)
        
    def fit_resample(self, X, y):
        n_maj = (y == 0).sum()
        if n_maj > 2000:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=self.random_state)
            X, y = rus.fit_resample(X, y)
        return self.svmsmote.fit_resample(X, y)

class SafeSMOTETomek:
    def __init__(self, random_state):
        self.random_state = random_state
        self.smotetomek = SMOTETomek(random_state=random_state)
        
    def fit_resample(self, X, y):
        # TomekLinks is O(N^2), creditcard will hang for days.
        n_maj = (y == 0).sum()
        if n_maj > 2000:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=self.random_state)
            X, y = rus.fit_resample(X, y)
        return self.smotetomek.fit_resample(X, y)

def get_resampler(name: str, random_state: int):
    """
    name in {'none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote', 'svmsmote', 'smotetomek'}
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
    elif name == 'svmsmote':
        return SafeSVMSMOTE(random_state=random_state)
    elif name == 'smotetomek':
        return SafeSMOTETomek(random_state=random_state)
    elif name == 'acsmote':
        from src.methods.ac_smote import ACSMOTE
        return ACSMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampler: {name}")

def get_classifier(name: str, random_state: int):
    if name == 'logreg':
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif name == 'rf':
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {name}")
