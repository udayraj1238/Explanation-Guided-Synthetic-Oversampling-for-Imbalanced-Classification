import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from src.utils import weighted_distance_matrix

class ACSMOTE:
    def __init__(self, sampling_strategy='auto', k_neighbors=5, alpha=2.0,
                 beta=0.7, surrogate_n_estimators=50, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.surrogate_n_estimators = surrogate_n_estimators
        self.random_state = random_state
        
        self.rng_ = np.random.RandomState(self.random_state)
        
    def _compute_w_hat(self, X_train, y_train, X_min):
        """Step 1: Surrogate model and attribution vector"""
        surrogate = RandomForestClassifier(
            n_estimators=self.surrogate_n_estimators, 
            max_depth=6,
            n_jobs=-1,
            random_state=self.random_state
        )
        surrogate.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X_min)
        
        # In newer shap versions for binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            # Assuming class 1 is the minority, we take shap_values[1]
            # but if shape is just (n, d), it's not a list. We'll handle both.
            # Usually RandomForest binary classification gives a list of length 2.
            # But minority label is always 1 in our setup.
            shap_values = shap_values[1]
            
        elif len(shap_values.shape) == 3:
            # SHAP returns (n_samples, n_features, n_classes) for some models
            shap_values = shap_values[:, :, 1]
            
        imp = np.mean(np.abs(shap_values), axis=0)
        
        min_imp = np.min(imp)
        max_imp = np.max(imp)
        
        if max_imp == min_imp:
            w_hat = np.full_like(imp, 0.5)
        else:
            w_hat = (imp - min_imp) / (max_imp - min_imp + 1e-12)
            
        return w_hat
        
    def _find_k_nearest(self, X_min, w_hat):
        """Step 2: Attribution-weighted neighbor search"""
        D = weighted_distance_matrix(X_min, w_hat, self.alpha)
        # Exclude self distance (diagonal)
        np.fill_diagonal(D, np.inf)
        # Find k nearest neighbors for each point
        nn_indices = np.argsort(D, axis=1)[:, :self.k_neighbors]
        return nn_indices
        
    def _generate_samples(self, X_min, nn_indices, w_hat, n_to_generate):
        """Step 3: Attribution-constrained interpolation"""
        n_min, d = X_min.shape
        synthetic_samples = np.zeros((n_to_generate, d))
        
        for idx in range(n_to_generate):
            i = self.rng_.choice(n_min)
            x_i = X_min[i]
            
            k_idx = self.rng_.choice(self.k_neighbors)
            nn_i = nn_indices[i, k_idx]
            x_nn = X_min[nn_i]
            
            lam = self.rng_.uniform(0, 1)
            
            lam_j = lam * (1.0 - self.beta * w_hat)
            lam_j = np.clip(lam_j, 0.0, 1.0)
            
            synthetic_samples[idx] = x_i + lam_j * (x_nn - x_i)
            
        return synthetic_samples
        
    def fit_resample(self, X, y):
        # Minority class is assumed to be 1 in our pipeline, majority 0
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        
        min_mask = y == 1
        maj_mask = y == 0
        
        X_min = X[min_mask]
        X_maj = X[maj_mask]
        
        n_min = len(X_min)
        n_maj = len(X_maj)
        
        if n_min == 0 or n_maj == 0 or n_min >= n_maj:
            return X.copy(), y.copy()
            
        n_to_generate = n_maj - n_min
        
        # Step 1
        w_hat = self._compute_w_hat(X, y, X_min)
        
        # Step 2
        nn_indices = self._find_k_nearest(X_min, w_hat)
        
        # Step 3
        synthetic_samples = self._generate_samples(X_min, nn_indices, w_hat, n_to_generate)
        
        X_res = np.vstack([X, synthetic_samples])
        y_res = np.concatenate([y, np.ones(n_to_generate, dtype=np.int64)])
        
        return X_res, y_res
