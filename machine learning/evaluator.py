"""
Cross-validation evaluator for CatBoost hyperparameters.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from data_utils import balance_data

RANDOM_STATE = 42

def evaluate_with_cv(params_array, X_data, y_data, n_folds=5):
    """Evaluate a hyperparameter set using 5-fold cross-validation."""
    params_dict = {
        'iterations': int(params_array[0]),
        'learning_rate': params_array[1],
        'depth': int(params_array[2]),
        'l2_leaf_reg': params_array[3],
        'border_count': int(params_array[4]),
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'auto_class_weights': 'Balanced',
        'random_state': RANDOM_STATE,
        'verbose': False,
        'use_best_model': False
    }
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X_data, y_data):
        X_tr, y_tr = balance_data(X_data.iloc[train_idx], y_data.iloc[train_idx])
        model = CatBoostClassifier(**params_dict)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_data.iloc[val_idx])
        scores.append(f1_score(y_data.iloc[val_idx], y_pred, zero_division=0))
    return np.mean(scores)