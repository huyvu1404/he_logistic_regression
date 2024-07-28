import numpy as np
from sklearn.metrics import roc_auc_score
from logistic_regression import LogisticRegression
from sklearn.model_selection import KFold

def grid_search_lr(X, y, param_grid):
    best_params = None
    best_score = 0

    for lr in param_grid['learning_rate']:
        for n_iter in param_grid['num_iterations']:
            for momen in param_grid['momentum']:
                model = LogisticRegression(learning_rate=lr, num_iterations=n_iter, momentum=momen)
  
                kf = KFold(n_splits=10, shuffle=True, random_state=1)
                auc_scores = []
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    y_pred_prob = model.predict_proba(X_test)
                    auc = roc_auc_score(y_test, y_pred_prob)
                    auc_scores.append(auc)

                score = np.mean(auc_scores)
                if score > best_score:
                    best_score = score
                    best_params = {'learning_rate': lr, 'num_iterations': n_iter, 'momentum': momen}

    return best_params, best_score
