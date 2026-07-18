"""
Unified interface for all baseline classifiers.
"""

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_classifier(name, **kwargs):
    """Return a classifier instance by name with library-default hyperparameters."""
    defaults = {
        'CatBoost': CatBoostClassifier(
            iterations=1000, learning_rate=0.1, depth=10,
            l2_leaf_reg=3, border_count=254,
            loss_function='Logloss', eval_metric='F1',
            auto_class_weights='Balanced', random_state=42, verbose=False
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.3,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, num_leaves=31, learning_rate=0.1,
            random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42, class_weight='balanced'
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', random_state=42, class_weight='balanced'
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100,), activation='relu', max_iter=200,
            random_state=42, early_stopping=True, validation_fraction=0.1
        )
    }
    if name not in defaults:
        raise ValueError(f"Unknown classifier: {name}")
    clf = defaults[name]
    if kwargs:
        clf.set_params(**kwargs)
    return clf