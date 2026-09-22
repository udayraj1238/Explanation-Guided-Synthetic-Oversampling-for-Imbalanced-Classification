import numpy as np
import random
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score

def weighted_distance_matrix(X_min: np.ndarray, w_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Returns the n_min x n_min matrix of d_w distances defined in Section 6.2 Step 2."""
    W = 1.0 + alpha * w_hat
    X_scaled = X_min * np.sqrt(W)
    X_sq = np.sum(X_scaled**2, axis=1)
    D_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * np.dot(X_scaled, X_scaled.T)
    D_sq = np.maximum(D_sq, 0.0)
    return np.sqrt(D_sq)

def compute_all_metrics(y_true, y_pred, y_score) -> dict:
    """Returns {'f1':..., 'auc_pr':..., 'mcc':..., 'gmean':..., 'bal_acc':...} per Section 8."""
    return {
        'f1': float(f1_score(y_true, y_pred, pos_label=1)),
        'auc_pr': float(average_precision_score(y_true, y_score)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'gmean': float(geometric_mean_score(y_true, y_pred)),
        'bal_acc': float(balanced_accuracy_score(y_true, y_pred))
    }

def set_all_seeds(seed: int) -> None:
    """Seeds numpy, python's random module, and anything else with its own RNG."""
    random.seed(seed)
    np.random.seed(seed)
