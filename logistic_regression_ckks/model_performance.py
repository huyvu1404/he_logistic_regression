from sklearn.metrics import  roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from homomorphic_encryption_functions import encrypt_label, encrypt_feature, encrypt_weights, decrypt_weights

import numpy as np


def sigmoid(x):
        return 0.5 + 0.197*x - 0.004*(x**3)

def predict(X, weights):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    linear_model = np.dot(X, weights) 
    proba = sigmoid(linear_model)
    predict_label = [1 if i > 0.5 else 0 for i in proba]
    return predict_label
    
def predict_proba(X, weights):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    linear_model = np.dot(X, weights)
    proba = sigmoid(linear_model)
    return proba 

def average_accuracy_and_auc_score_helr(model, n, X, y, encryptor, decryptor, ckks_encoder, scale, evaluator, relin_keys, galois_keys, slot_count):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    auc_scores = []
    acc_scores = []
   
    for train_index, test_index in kf.split(X):
      
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_ctx = encrypt_feature(X_train, encryptor, ckks_encoder, scale, slot_count)
        y_ctx = encrypt_label(y_train, encryptor, ckks_encoder, scale)

        w = [0] * len(X_ctx)
        w_ctx  = encrypt_weights(w, ckks_encoder, scale, encryptor)
        v_ctx  = w_ctx
        for _ in range(n):
            new_v_ctx, new_w_ctx = model.fit(X_ctx, y_ctx, w_ctx, v_ctx,ckks_encoder, scale, evaluator, relin_keys, galois_keys, slot_count)
            new_v = decrypt_weights(new_v_ctx, ckks_encoder, decryptor)
            new_w = decrypt_weights(new_w_ctx, ckks_encoder, decryptor)

            re_encrypted_w = encrypt_weights(new_w, ckks_encoder, scale, encryptor)
            re_encrypted_v = encrypt_weights(new_v, ckks_encoder, scale, encryptor)
            w_ctx = re_encrypted_w
            v_ctx = re_encrypted_v
        new_w = decrypt_weights(new_w_ctx, ckks_encoder, decryptor)
        y_prob = predict_proba(X_test, new_w)
        y_pred = predict(X_test, new_w)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        acc_scores.append(acc)
        auc_scores.append(auc)    
   

    average_acc = np.mean(acc_scores)
    average_auc = np.mean(auc_scores)
    
    return average_acc, average_auc
