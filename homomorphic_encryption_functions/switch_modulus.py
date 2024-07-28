def switch_cipher_modulus(arr_ciphertext, params, evaluator):
    
    switched_arr_ciphertext = []
    for i in range(len(arr_ciphertext)):
        switched_ciphertext = evaluator.mod_switch_to(arr_ciphertext[i], params)
        switched_arr_ciphertext.append(switched_ciphertext)
        
    return switched_arr_ciphertext