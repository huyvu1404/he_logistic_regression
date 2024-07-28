from sklearn.metrics import  roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import numpy as np

def average_accuracy_and_auc_score(model, X, y):

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    acc_scores = []
    auc_scores = []
 
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        acc_scores.append(acc)
        auc_scores.append(auc)
     
    average_acc = np.mean(acc_scores) 
    average_auc = np.mean(auc_scores)
    
    return average_acc, average_auc

