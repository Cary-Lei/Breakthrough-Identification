"""
PSO optimizer.
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

def pso_optimize(X_train_val, y_train_val, X_test, y_test,
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
    vel = np.random.uniform(-0.1, 0.1, pop.shape)

    fitness = evaluate_population(pop, X_train_val, y_train_val, n_folds)

    pbest = pop.copy()
    pbest_fit = fitness.copy()
    gbest = pop[np.argmax(fitness)].copy()
    gbest_fit = np.max(fitness)

    history = {'best': [gbest_fit], 'params': [gbest.copy()]}

    w, c1, c2 = 0.7, 1.5, 1.5

    for t in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.random(5), np.random.random(5)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])

            new_pos = pop[i] + vel[i]
            new_pos[0] = np.clip(new_pos[0], *param_ranges['iterations'])
            new_pos[1] = np.clip(new_pos[1], *param_ranges['learning_rate'])
            new_pos[2] = np.clip(new_pos[2], *param_ranges['depth'])
            new_pos[3] = np.clip(new_pos[3], *param_ranges['l2_leaf_reg'])
            new_pos[4] = np.clip(new_pos[4], *param_ranges['border_count'])
            new_pos[[0, 2, 4]] = new_pos[[0, 2, 4]].astype(int)

            new_fit = evaluate_with_cv(new_pos, X_train_val, y_train_val, n_folds)

            if new_fit > fitness[i]:
                pop[i], fitness[i] = new_pos, new_fit
                if new_fit > pbest_fit[i]:
                    pbest[i], pbest_fit[i] = new_pos, new_fit
                    if new_fit > gbest_fit:
                        gbest, gbest_fit = new_pos, new_fit

        history['best'].append(gbest_fit)
        history['params'].append(gbest.copy())

    return gbest, gbest_fit, history, param_names