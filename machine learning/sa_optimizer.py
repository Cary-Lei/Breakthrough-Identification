"""
Simulated Annealing optimizer.
"""

import numpy as np
from evaluator import evaluate_with_cv

MAX_ITER = 200

def sa_optimize(X_train_val, y_train_val, X_test, y_test, max_iter=MAX_ITER, n_folds=5):
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score
    from data_utils import balance_data

    param_ranges = {
        'iterations': (500, 1500),
        'learning_rate': (0.05, 0.3),
        'depth': (6, 12),
        'l2_leaf_reg': (1, 5),
        'border_count': (200, 300),
    }
    param_names = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'border_count']

    current = np.array([1000, 0.1, 10, 3, 254])
    current_fit = evaluate_with_cv(current, X_train_val, y_train_val, n_folds)

    best = current.copy()
    best_fit = current_fit

    history = {'best': [best_fit], 'params': [best.copy()]}

    for t in range(max_iter):
        neighbor = current.copy()
        idx = np.random.randint(5)
        if idx == 0:
            neighbor[idx] += np.random.randint(-100, 100)
        elif idx == 1:
            neighbor[idx] += np.random.uniform(-0.02, 0.02)
        elif idx == 2:
            neighbor[idx] += np.random.randint(-1, 2)
        elif idx == 3:
            neighbor[idx] += np.random.uniform(-0.5, 0.5)
        elif idx == 4:
            neighbor[idx] += np.random.randint(-10, 10)

        neighbor[0] = np.clip(neighbor[0], *param_ranges['iterations'])
        neighbor[1] = np.clip(neighbor[1], *param_ranges['learning_rate'])
        neighbor[2] = np.clip(neighbor[2], *param_ranges['depth'])
        neighbor[3] = np.clip(neighbor[3], *param_ranges['l2_leaf_reg'])
        neighbor[4] = np.clip(neighbor[4], *param_ranges['border_count'])
        neighbor[[0, 2, 4]] = neighbor[[0, 2, 4]].astype(int)

        neighbor_fit = evaluate_with_cv(neighbor, X_train_val, y_train_val, n_folds)

        T = max(0.01, 1 - t / max_iter)
        if neighbor_fit > current_fit or np.random.random() < np.exp((neighbor_fit - current_fit) / T):
            current, current_fit = neighbor, neighbor_fit

        if current_fit > best_fit:
            best, best_fit = current.copy(), current_fit

        history['best'].append(best_fit)
        history['params'].append(best.copy())

    return best, best_fit, history, param_names