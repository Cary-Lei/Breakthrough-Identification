"""
Optimizer comparison across all classifiers (Figure 5).
Runs L-SABO, SABO, PSO, SA on every classifier.
All parameters are numeric (no discrete/categorical choices).
Paper parameters: POP_SIZE=50, MAX_ITER=200, N_FOLDS=5.
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import joblib
from scipy.special import gamma

from data_utils import load_data, balance_data
from classifiers import get_classifier
from sabo_optimizer import sabo_optimize
from lsabo_optimizer import lsabo_optimize
from pso_optimizer import pso_optimize
from sa_optimizer import sa_optimize

DATA_PATH = "data/final_dataset.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
POP_SIZE = 50
MAX_ITER = 200


def optimize_classifier(clf_name, X_train_val, y_train_val, X_test, y_test, opt_name, opt_func):
    """Run a specific optimizer on a specific classifier."""
    print(f"  Running {opt_name} on {clf_name}...")
    start = time.time()

    if clf_name == 'CatBoost':
        if opt_name == 'SA':
            _, best_f1, _, _ = opt_func(X_train_val, y_train_val, X_test, y_test,
                                         max_iter=MAX_ITER, n_folds=N_FOLDS)
        else:
            _, best_f1, _, _ = opt_func(X_train_val, y_train_val, X_test, y_test,
                                         pop_size=POP_SIZE, max_iter=MAX_ITER, n_folds=N_FOLDS)
    else:
        best_f1 = optimize_generic_classifier(clf_name, X_train_val, y_train_val, X_test, y_test,
                                               opt_name, opt_func)

    elapsed = time.time() - start
    print(f"    {opt_name} F1: {best_f1:.4f} (took {elapsed:.2f}s)")
    return best_f1


def optimize_generic_classifier(clf_name, X_train_val, y_train_val, X_test, y_test,
                                 opt_name, opt_func):
    """Generic optimizer wrapper for non-CatBoost classifiers (all numeric params)."""
    param_spaces = {
        'XGBoost': {
            'n_estimators': (500, 1500),
            'max_depth': (6, 12),
            'learning_rate': (0.05, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
        },
        'LightGBM': {
            'n_estimators': (500, 1500),
            'num_leaves': (20, 50),
            'learning_rate': (0.05, 0.3),
            'min_child_samples': (10, 30),
            'subsample': (0.6, 1.0),
        },
        'RandomForest': {
            'n_estimators': (500, 1500),
            'max_depth': (6, 12),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5),
            'max_features': (0.3, 1.0),
        },
        'SVM': {
            'C': (0.1, 10.0),
            'gamma': (0.001, 0.1),
            'degree': (2, 4),
            'coef0': (0.0, 1.0),
            'tol': (1e-5, 1e-3),
        },
        'MLP': {
            'hidden_layer_sizes_1': (50, 200),
            'hidden_layer_sizes_2': (0, 100),
            'alpha': (0.0001, 0.01),
            'learning_rate_init': (0.001, 0.01),
            'tol': (1e-5, 1e-3),
        }
    }

    if clf_name not in param_spaces:
        clf = get_classifier(clf_name)
        X_bal, y_bal = balance_data(X_train_val, y_train_val)
        clf.fit(X_bal, y_bal)
        return f1_score(y_test, clf.predict(X_test), zero_division=0)

    param_names = list(param_spaces[clf_name].keys())
    param_ranges = param_spaces[clf_name]

    if opt_name == 'SABO':
        return sabo_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                                      X_test, y_test, clf_name)
    elif opt_name == 'L-SABO':
        return lsabo_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                                       X_test, y_test, clf_name)
    elif opt_name == 'PSO':
        return pso_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                                     X_test, y_test, clf_name)
    elif opt_name == 'SA':
        return sa_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                                    X_test, y_test, clf_name)
    else:
        clf = get_classifier(clf_name)
        X_bal, y_bal = balance_data(X_train_val, y_train_val)
        clf.fit(X_bal, y_bal)
        return f1_score(y_test, clf.predict(X_test), zero_division=0)


def evaluate_generic(params, param_names, param_ranges, clf_name, X_train, y_train, X_val, y_val):
    """Evaluate a parameter set for a generic classifier."""
    params_dict = {}
    for i, name in enumerate(param_names):
        if name == 'hidden_layer_sizes_1':
            h1 = int(params[i])
            h2_idx = param_names.index('hidden_layer_sizes_2') if 'hidden_layer_sizes_2' in param_names else -1
            h2 = int(params[h2_idx]) if h2_idx != -1 and h2_idx < len(params) else 0
            if h2 > 0:
                params_dict['hidden_layer_sizes'] = (h1, h2)
            else:
                params_dict['hidden_layer_sizes'] = (h1,)
        elif name == 'hidden_layer_sizes_2':
            continue
        elif name in ['n_estimators', 'num_leaves', 'max_depth',
                      'min_samples_split', 'min_samples_leaf', 'degree']:
            params_dict[name] = int(params[i])
        else:
            params_dict[name] = params[i]

    X_bal, y_bal = balance_data(X_train, y_train)
    clf = get_classifier(clf_name, **params_dict)
    clf.fit(X_bal, y_bal)
    return f1_score(y_val, clf.predict(X_val), zero_division=0)


def sabo_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                          X_test, y_test, clf_name):
    pop_size = POP_SIZE
    max_iter = MAX_ITER
    n_folds = N_FOLDS

    pop = []
    for _ in range(pop_size):
        ind = []
        for name in param_names:
            if name == 'hidden_layer_sizes_2':
                continue
            low, high = param_ranges[name]
            if name in ['n_estimators', 'num_leaves', 'max_depth',
                        'min_samples_split', 'min_samples_leaf', 'degree']:
                ind.append(np.random.randint(low, high + 1))
            else:
                ind.append(np.random.uniform(low, high))
        pop.append(np.array(ind))
    pop = np.array(pop, dtype=object)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def evaluate_cv(ind):
        scores = []
        for train_idx, val_idx in skf.split(X_train_val, y_train_val):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            score = evaluate_generic(ind, param_names, param_ranges, clf_name,
                                      X_tr, y_tr, X_val, y_val)
            scores.append(score)
        return np.mean(scores)

    fitness_cv = np.array([evaluate_cv(ind) for ind in pop])

    best_idx = np.argmax(fitness_cv)
    best_sol = pop[best_idx].copy()
    best_cv = fitness_cv[best_idx]

    patience, no_improve = 10, 0

    for t in range(max_iter):
        for i in range(pop_size):
            d = np.zeros_like(pop[i], dtype=float)
            for j in range(pop_size):
                if i != j:
                    v = np.random.choice([1, 2])
                    d += np.sign(fitness_cv[i] - fitness_cv[j]) * (pop[i].astype(float) - v * pop[j].astype(float))

            new_pos = pop[i].astype(float) + d / pop_size

            for idx, name in enumerate(param_names):
                if name == 'hidden_layer_sizes_2':
                    continue
                low, high = param_ranges[name]
                if name in ['n_estimators', 'num_leaves', 'max_depth',
                            'min_samples_split', 'min_samples_leaf', 'degree']:
                    new_pos[idx] = int(np.clip(new_pos[idx], low, high))
                else:
                    new_pos[idx] = np.clip(new_pos[idx], low, high)

            new_fitness = evaluate_cv(new_pos)

            if new_fitness > fitness_cv[i] + 0.001:
                pop[i] = new_pos
                fitness_cv[i] = new_fitness

                if new_fitness > best_cv + 0.001:
                    best_cv = new_fitness
                    best_sol = new_pos.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        if (t + 1) % 10 == 0:
            worst = np.argmin(fitness_cv)
            new_ind = []
            for name in param_names:
                if name == 'hidden_layer_sizes_2':
                    continue
                low, high = param_ranges[name]
                if name in ['n_estimators', 'num_leaves', 'max_depth',
                            'min_samples_split', 'min_samples_leaf', 'degree']:
                    new_ind.append(np.random.randint(low, high + 1))
                else:
                    new_ind.append(np.random.uniform(low, high))
            pop[worst] = np.array(new_ind)
            fitness_cv[worst] = evaluate_cv(pop[worst])

        if no_improve >= patience:
            break

    final_f1 = evaluate_generic(best_sol, param_names, param_ranges, clf_name,
                                 X_train_val, y_train_val, X_test, y_test)
    return final_f1


def lsabo_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                            X_test, y_test, clf_name):
    pop_size = POP_SIZE
    max_iter = MAX_ITER
    n_folds = N_FOLDS

    pop = []
    for _ in range(pop_size):
        ind = []
        for name in param_names:
            if name == 'hidden_layer_sizes_2':
                continue
            low, high = param_ranges[name]
            if name in ['n_estimators', 'num_leaves', 'max_depth',
                        'min_samples_split', 'min_samples_leaf', 'degree']:
                ind.append(np.random.randint(low, high + 1))
            else:
                ind.append(np.random.uniform(low, high))
        pop.append(np.array(ind))
    pop = np.array(pop, dtype=object)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def evaluate_cv(ind):
        scores = []
        for train_idx, val_idx in skf.split(X_train_val, y_train_val):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            score = evaluate_generic(ind, param_names, param_ranges, clf_name,
                                      X_tr, y_tr, X_val, y_val)
            scores.append(score)
        return np.mean(scores)

    fitness_cv = np.array([evaluate_cv(ind) for ind in pop])

    best_idx = np.argmax(fitness_cv)
    best_sol = pop[best_idx].copy()
    best_cv = fitness_cv[best_idx]

    patience, no_improve = 10, 0

    for t in range(max_iter):
        for i in range(pop_size):
            d = np.zeros_like(pop[i], dtype=float)
            for j in range(pop_size):
                if i != j:
                    v = np.random.choice([1, 2])
                    d += np.sign(fitness_cv[i] - fitness_cv[j]) * (pop[i].astype(float) - v * pop[j].astype(float))

            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                     (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            levy = 0.01 * sigma * np.random.normal(0, 1) / (abs(np.random.normal(0, 1)) ** (1 / beta))

            new_pos = pop[i].astype(float) + levy * d / pop_size

            for idx, name in enumerate(param_names):
                if name == 'hidden_layer_sizes_2':
                    continue
                low, high = param_ranges[name]
                if name in ['n_estimators', 'num_leaves', 'max_depth',
                            'min_samples_split', 'min_samples_leaf', 'degree']:
                    new_pos[idx] = int(np.clip(new_pos[idx], low, high))
                else:
                    new_pos[idx] = np.clip(new_pos[idx], low, high)

            new_fitness = evaluate_cv(new_pos)

            if new_fitness > fitness_cv[i] + 0.001:
                pop[i] = new_pos
                fitness_cv[i] = new_fitness

                if new_fitness > best_cv + 0.001:
                    best_cv = new_fitness
                    best_sol = new_pos.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        if (t + 1) % 10 == 0:
            worst = np.argmin(fitness_cv)
            new_ind = []
            for name in param_names:
                if name == 'hidden_layer_sizes_2':
                    continue
                low, high = param_ranges[name]
                if name in ['n_estimators', 'num_leaves', 'max_depth',
                            'min_samples_split', 'min_samples_leaf', 'degree']:
                    new_ind.append(np.random.randint(low, high + 1))
                else:
                    new_ind.append(np.random.uniform(low, high))
            pop[worst] = np.array(new_ind)
            fitness_cv[worst] = evaluate_cv(pop[worst])

        if no_improve >= patience:
            break

    final_f1 = evaluate_generic(best_sol, param_names, param_ranges, clf_name,
                                 X_train_val, y_train_val, X_test, y_test)
    return final_f1


def pso_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                          X_test, y_test, clf_name):
    pop_size = POP_SIZE
    max_iter = MAX_ITER
    n_folds = N_FOLDS

    pop = []
    for _ in range(pop_size):
        ind = []
        for name in param_names:
            if name == 'hidden_layer_sizes_2':
                continue
            low, high = param_ranges[name]
            if name in ['n_estimators', 'num_leaves', 'max_depth',
                        'min_samples_split', 'min_samples_leaf', 'degree']:
                ind.append(np.random.randint(low, high + 1))
            else:
                ind.append(np.random.uniform(low, high))
        pop.append(np.array(ind))
    pop = np.array(pop, dtype=object)
    vel = np.random.uniform(-0.1, 0.1, pop.shape).astype(object)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def evaluate_cv(ind):
        scores = []
        for train_idx, val_idx in skf.split(X_train_val, y_train_val):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            score = evaluate_generic(ind, param_names, param_ranges, clf_name,
                                      X_tr, y_tr, X_val, y_val)
            scores.append(score)
        return np.mean(scores)

    fitness = np.array([evaluate_cv(ind) for ind in pop])

    pbest = pop.copy()
    pbest_fit = fitness.copy()
    gbest = pop[np.argmax(fitness)].copy()
    gbest_fit = np.max(fitness)

    w, c1, c2 = 0.7, 1.5, 1.5

    for t in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.random(len(param_names)), np.random.random(len(param_names))
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])

            new_pos = pop[i] + vel[i]

            for idx, name in enumerate(param_names):
                if name == 'hidden_layer_sizes_2':
                    continue
                low, high = param_ranges[name]
                if name in ['n_estimators', 'num_leaves', 'max_depth',
                            'min_samples_split', 'min_samples_leaf', 'degree']:
                    new_pos[idx] = int(np.clip(new_pos[idx], low, high))
                else:
                    new_pos[idx] = np.clip(new_pos[idx], low, high)

            new_fit = evaluate_cv(new_pos)

            if new_fit > fitness[i]:
                pop[i] = new_pos
                fitness[i] = new_fit
                if new_fit > pbest_fit[i]:
                    pbest[i] = new_pos
                    pbest_fit[i] = new_fit
                    if new_fit > gbest_fit:
                        gbest = new_pos
                        gbest_fit = new_fit

    final_f1 = evaluate_generic(gbest, param_names, param_ranges, clf_name,
                                 X_train_val, y_train_val, X_test, y_test)
    return final_f1


def sa_optimize_generic(param_names, param_ranges, X_train_val, y_train_val,
                         X_test, y_test, clf_name):
    max_iter = MAX_ITER
    n_folds = N_FOLDS

    current = []
    for name in param_names:
        if name == 'hidden_layer_sizes_2':
            continue
        low, high = param_ranges[name]
        if name in ['n_estimators', 'num_leaves', 'max_depth',
                    'min_samples_split', 'min_samples_leaf', 'degree']:
            current.append(np.random.randint(low, high + 1))
        else:
            current.append(np.random.uniform(low, high))
    current = np.array(current)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def evaluate_cv(ind):
        scores = []
        for train_idx, val_idx in skf.split(X_train_val, y_train_val):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            score = evaluate_generic(ind, param_names, param_ranges, clf_name,
                                      X_tr, y_tr, X_val, y_val)
            scores.append(score)
        return np.mean(scores)

    current_fit = evaluate_cv(current)

    best = current.copy()
    best_fit = current_fit

    for t in range(max_iter):
        neighbor = current.copy()
        idx = np.random.randint(len(param_names))
        name = param_names[idx]

        if name == 'hidden_layer_sizes_2':
            continue

        if name in ['n_estimators', 'num_leaves', 'max_depth',
                    'min_samples_split', 'min_samples_leaf', 'degree']:
            step = max(1, int((param_ranges[name][1] - param_ranges[name][0]) / 20))
            neighbor[idx] += np.random.randint(-step, step + 1)
            neighbor[idx] = np.clip(neighbor[idx], param_ranges[name][0], param_ranges[name][1])
            neighbor[idx] = int(neighbor[idx])
        else:
            step = (param_ranges[name][1] - param_ranges[name][0]) / 20
            neighbor[idx] += np.random.uniform(-step, step)
            neighbor[idx] = np.clip(neighbor[idx], param_ranges[name][0], param_ranges[name][1])

        neighbor_fit = evaluate_cv(neighbor)

        T = max(0.01, 1 - t / max_iter)
        if neighbor_fit > current_fit or np.random.random() < np.exp((neighbor_fit - current_fit) / T):
            current = neighbor
            current_fit = neighbor_fit

        if current_fit > best_fit:
            best = current.copy()
            best_fit = current_fit

    final_f1 = evaluate_generic(best, param_names, param_ranges, clf_name,
                                 X_train_val, y_train_val, X_test, y_test)
    return final_f1


def run_optimizers():
    X, y = load_data(DATA_PATH)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'SVM', 'MLP']
    opt_funcs = {
        'L-SABO': lsabo_optimize,
        'SABO': sabo_optimize,
        'PSO': pso_optimize,
        'SA': sa_optimize,
    }
    results = {clf: {} for clf in classifiers}

    for clf_name in classifiers:
        print(f"\n{'='*60}")
        print(f"Classifier: {clf_name}")
        print('='*60)

        base_clf = get_classifier(clf_name)
        if clf_name != 'CatBoost':
            X_bal, y_bal = balance_data(X_train_val, y_train_val)
            base_clf.fit(X_bal, y_bal)
        else:
            base_clf.fit(X_train_val, y_train_val)
        baseline_f1 = f1_score(y_test, base_clf.predict(X_test), zero_division=0)
        results[clf_name]['Baseline'] = baseline_f1
        print(f"Baseline F1: {baseline_f1:.4f}")

        for opt_name, opt_func in opt_funcs.items():
            best_f1 = optimize_classifier(clf_name, X_train_val, y_train_val, X_test, y_test,
                                           opt_name, opt_func)
            results[clf_name][opt_name] = best_f1

    print("\n" + "=" * 70)
    print("Figure 5 Results: All Classifiers × All Optimizers")
    print("=" * 70)

    print(f"{'Classifier':<15}", end="")
    for opt in ['Baseline', 'L-SABO', 'SABO', 'PSO', 'SA']:
        print(f"{opt:<12}", end="")
    print()

    for clf_name in classifiers:
        print(f"{clf_name:<15}", end="")
        for opt in ['Baseline', 'L-SABO', 'SABO', 'PSO', 'SA']:
            val = results[clf_name].get(opt)
            if val is None:
                print(f"{'N/A':<12}", end="")
            else:
                print(f"{val:.4f}{'':<8}", end="")
        print()

    print("\nSaving CatBoost model for SHAP analysis...")
    sample_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.1, depth=10,
        l2_leaf_reg=3, border_count=254,
        loss_function='Logloss', eval_metric='F1',
        auto_class_weights='Balanced', random_state=42, verbose=False
    )
    X_bal, y_bal = balance_data(X_train_val, y_train_val)
    sample_model.fit(X_bal, y_bal)
    joblib.dump({'model': sample_model, 'feature_names': list(X.columns)},
                'final_optimized_catboost_model.pkl')
    print("Model saved to final_optimized_catboost_model.pkl")


if __name__ == "__main__":
    run_optimizers()