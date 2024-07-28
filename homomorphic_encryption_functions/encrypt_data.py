import numpy as np

def encrypt_feature(df, encryptor, ckks_encoder, scale, slot_count):
    
    num_observations, num_columns = df.shape
    slot_null = [0]*(slot_count - num_observations)
    
    bias_ptx = ckks_encoder.encode(np.concatenate(([1]*num_observations, slot_null)), scale)
    bias_ctx = encryptor.encrypt(bias_ptx)
    encrypted_data = [bias_ctx]
    
    # Apply batch encode to each column
    for i in range(num_columns):
        feature = df[:, i]
        
        feature_ptx = ckks_encoder.encode(np.concatenate((feature, slot_null)), scale)
        feature_ctx = encryptor.encrypt(feature_ptx)
            
        encrypted_data.append(feature_ctx)
    return encrypted_data

def encrypt_label(df, encryptor, ckks_encoder, scale):

    df_ptx  = ckks_encoder.encode(df, scale)
    df_ctx = encryptor.encrypt(df_ptx)
        
    return df_ctx

def encrypt_weights(weights, ckks_encoder, scale, encryptor):
    encrypted_weights = []
    for weight in weights:
        weight_ptx = ckks_encoder.encode([weight] * ckks_encoder.slot_count(), scale)
        weight_ctx = encryptor.encrypt(weight_ptx)
        encrypted_weights.append(weight_ctx)
    
    return encrypted_weights

def prepare_encrypted_data(X_train, y_train, encryptor, ckks_encoder, scale, slot_count):
    X_train_ctx = encrypt_feature(X_train, encryptor, ckks_encoder, scale, slot_count)
    y_train_ctx = encrypt_label(y_train, encryptor, ckks_encoder, scale)
    return X_train_ctx, y_train_ctx