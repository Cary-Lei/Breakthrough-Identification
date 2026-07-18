"""
SABO optimizer without Lévy flight.
"""

import numpy as np
from evaluator import evaluate_with_cv

POP_SIZE = 50
MAX_ITER = 200

def evaluate_population(population, X_train_val, y_train_val, n_folds=5):
    fitness = np.zeros(len(population))
    for i, ind in enumerate(population):
        fitness[i] = evaluate_with_cv(ind, X_train_val, y_train_val, n_folds)
    return fitness

def sabo_optimize(X_train_val, y_train_val, X_test, y_test,
                  pop_size=POP_SIZE, max_iter=MAX_ITER, n_folds=5):
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

    pop = [np.array([1000, 0.1, 10, 3, 254])]
    for _ in range(1, pop_size):
        pop.append(np.array([
            np.random.randint(*param_ranges['iterations']),
            np.random.uniform(*param_ranges['learning_rate']),
            np.random.randint(*param_ranges['depth']),
            np.random.uniform(*param_ranges['l2_leaf_reg']),
            np.random.randint(*param_ranges['border_count']),
        ]))
    pop = np.array(pop)

    fitness_cv = evaluate_population(pop, X_train_val, y_train_val, n_folds)

    fitness_test = []
    for ind in pop:
        X_bal, y_bal = balance_data(X_train_val, y_train_val)
        model = CatBoostClassifier(
            iterations=int(ind[0]), learning_rate=ind[1], depth=int(ind[2]),
            l2_leaf_reg=ind[3], border_count=int(ind[4]),
            loss_function='Logloss', eval_metric='F1',
            auto_class_weights='Balanced', random_state=42, verbose=False
        )
        model.fit(X_bal, y_bal)
        fitness_test.append(f1_score(y_test, model.predict(X_test), zero_division=0))
    fitness_test = np.array(fitness_test)

    best_idx = np.argmax(fitness_cv)
    best_sol = pop[best_idx].copy()
    best_cv = fitness_cv[best_idx]
    best_test = fitness_test[best_idx]

    history = {'best_cv': [best_cv], 'best_test': [best_test], 'params': [best_sol.copy()]}

    patience, no_improve = 10, 0

    for t in range(max_iter):
        for i in range(pop_size):
            d = np.zeros_like(pop[i])
            for j in range(pop_size):
                if i != j:
                    v = np.random.choice([1, 2])
                    d += np.sign(fitness_cv[i] - fitness_cv[j]) * (pop[i] - v * pop[j])

            new_pos = pop[i] + d / pop_size
            new_pos[0] = np.clip(new_pos[0], *param_ranges['iterations'])
            new_pos[1] = np.clip(new_pos[1], *param_ranges['learning_rate'])
            new_pos[2] = np.clip(new_pos[2], *param_ranges['depth'])
            new_pos[3] = np.clip(new_pos[3], *param_ranges['l2_leaf_reg'])
            new_pos[4] = np.clip(new_pos[4], *param_ranges['border_count'])
            new_pos[[0, 2, 4]] = new_pos[[0, 2, 4]].astype(int)

            new_fitness = evaluate_with_cv(new_pos, X_train_val, y_train_val, n_folds)

            if new_fitness > fitness_cv[i] + 0.001:
                pop[i], fitness_cv[i] = new_pos, new_fitness
                X_bal, y_bal = balance_data(X_train_val, y_train_val)
                model = CatBoostClassifier(
                    iterations=int(new_pos[0]), learning_rate=new_pos[1], depth=int(new_pos[2]),
                    l2_leaf_reg=new_pos[3], border_count=int(new_pos[4]),
                    loss_function='Logloss', eval_metric='F1',
                    auto_class_weights='Balanced', random_state=42, verbose=False
                )
                model.fit(X_bal, y_bal)
                fitness_test[i] = f1_score(y_test, model.predict(X_test), zero_division=0)

                if new_fitness > best_cv + 0.001:
                    best_cv, best_test, best_sol = new_fitness, fitness_test[i], new_pos.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        if (t + 1) % 10 == 0:
            worst = np.argmin(fitness_cv)
            pop[worst] = np.array([
                np.random.randint(*param_ranges['iterations']),
                np.random.uniform(*param_ranges['learning_rate']),
                np.random.randint(*param_ranges['depth']),
                np.random.uniform(*param_ranges['l2_leaf_reg']),
                np.random.randint(*param_ranges['border_count']),
            ])
            fitness_cv[worst] = evaluate_with_cv(pop[worst], X_train_val, y_train_val, n_folds)

        history['best_cv'].append(best_cv)
        history['best_test'].append(best_test)
        history['params'].append(best_sol.copy())

        if no_improve >= patience:
            break

    return best_sol, best_cv, best_test, history, param_names