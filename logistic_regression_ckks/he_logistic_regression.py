from homomorphic_encryption_functions import dot_product_ciphertexts, eval_poly, sum_slots, switch_cipher_modulus


class HELogisticRegression():   
    
    def __init__(self, learning_rate, momentum = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
         
    def fit(self, X_ctx, y_ctx, w_ctx, v_ctx, ckks_encoder, scale, evaluator, relin_keys, galois_keys, slot_count):
            
            m_ptx = ckks_encoder.encode([self.momentum]*slot_count, scale)    
            lookahead_weights = []
            momen_volocity_arr = []
            for i in range(len(v_ctx)):
                momen_volocity = evaluator.multiply_plain(v_ctx[i], m_ptx)  
                evaluator.relinearize_inplace(momen_volocity, relin_keys)
                evaluator.rescale_to_next_inplace(momen_volocity)
                momen_volocity.scale(scale)
                momen_volocity_arr.append(momen_volocity)  
                 
                switch_weight = evaluator.mod_switch_to(w_ctx[i], momen_volocity.parms_id())
                lookahead_weight = evaluator.add(switch_weight, momen_volocity)  
                lookahead_weights.append(lookahead_weight) 
            
            switched_X_ctx = switch_cipher_modulus(X_ctx, lookahead_weights[i].parms_id(), evaluator)
            X_W = dot_product_ciphertexts(switched_X_ctx, lookahead_weights, scale, evaluator, relin_keys) 
            y_pred = eval_poly(X_W, ckks_encoder, scale, evaluator, relin_keys) 
            switched_y_ctx = evaluator.mod_switch_to(y_ctx, y_pred.parms_id()) 
            diff = evaluator.sub(y_pred, switched_y_ctx) 
        
            new_w_ctx = []
            new_v_ctx = [] 
  
            for i in range(len(X_ctx)):
                
                switched_X = evaluator.mod_switch_to(X_ctx[i], diff.parms_id())  
                dw = dot_product_ciphertexts([diff], [switched_X], scale, evaluator, relin_keys) 
                sumslots_dw = sum_slots(dw, evaluator, galois_keys, slot_count) 
                
                lr_ptx = ckks_encoder.encode([self.learning_rate], scale)
                switched_lr_ptx = evaluator.mod_switch_to(lr_ptx, sumslots_dw.parms_id())
                grad = evaluator.multiply_plain(sumslots_dw, switched_lr_ptx)  
                evaluator.relinearize_inplace(grad, relin_keys)
                evaluator.rescale_to_next_inplace(grad)
                grad.scale(scale)

                switched_m_v = evaluator.mod_switch_to(momen_volocity_arr[i], grad.parms_id()) 
                switched_w = evaluator.mod_switch_to(w_ctx[i], grad.parms_id())
                
                new_v = evaluator.sub(switched_m_v, grad)
                new_v_ctx.append(new_v) 
                new_w = evaluator.add(switched_w, new_v) 
                new_w_ctx.append(new_w)
            
            
            return new_v_ctx, new_w_ctx