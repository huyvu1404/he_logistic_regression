def decrypt_weights(encrypted_weights, ckks_encoder, decryptor):
    weights = []
    for weight_ctx in encrypted_weights:
        weight_ptx = decryptor.decrypt(weight_ctx)
        weight = ckks_encoder.decode(weight_ptx)

        weights.append(weight[0])
            
    return weights