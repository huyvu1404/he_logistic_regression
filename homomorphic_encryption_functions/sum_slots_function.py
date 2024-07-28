import numpy as np

def sum_slots(encrypted_vector, evaluator, galois_keys, slot_count):

    for i in range(int(np.log2(slot_count))):
        rotated = evaluator.rotate_vector(encrypted_vector, 2**i, galois_keys)
        encrypted_vector = evaluator.add(encrypted_vector, rotated)
        
    return encrypted_vector