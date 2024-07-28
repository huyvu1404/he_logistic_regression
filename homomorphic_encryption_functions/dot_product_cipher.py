import numpy as np

def dot_product_ciphertexts(encrypted_A, encrypted_B, scale, evaluator, relin_keys):
    size = len(encrypted_A)
    result = evaluator.multiply(encrypted_A[0], encrypted_B[0])
    evaluator.relinearize_inplace(result, relin_keys)
    
    for i in range(1, size):
        temp = evaluator.multiply(encrypted_A[i], encrypted_B[i])
        evaluator.relinearize_inplace(temp, relin_keys)
        result = evaluator.add(result, temp)
       
    evaluator.rescale_to_next_inplace(result)
    result.scale(scale)
    return result


