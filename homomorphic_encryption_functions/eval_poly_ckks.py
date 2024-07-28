def eval_poly(encrypted_vector, ckks_encoder, scale, evaluator, relin_keys):
    
    c = ckks_encoder.encode(-0.004, scale)
    b = ckks_encoder.encode(0.197, scale)
    a = ckks_encoder.encode(0.5, scale)
    
    x2_ctx = evaluator.square(encrypted_vector)
    evaluator.relinearize_inplace(x2_ctx, relin_keys)

    evaluator.rescale_to_next_inplace(x2_ctx)

    evaluator.mod_switch_to_inplace(c, encrypted_vector.parms_id())
    c_x_ctx = evaluator.multiply_plain(encrypted_vector, c)
    
    evaluator.rescale_to_next_inplace(c_x_ctx)

    c_x3_ctx = evaluator.multiply(x2_ctx, c_x_ctx)
    evaluator.relinearize_inplace(c_x3_ctx, relin_keys)
    evaluator.rescale_to_next_inplace(c_x3_ctx)

    evaluator.mod_switch_to_inplace(b, encrypted_vector.parms_id())
    b_x_ctx = evaluator.multiply_plain(encrypted_vector, b)
    evaluator.rescale_to_next_inplace(b_x_ctx)

    c_x3_ctx.scale(pow(2.0, 50))
    b_x_ctx.scale(pow(2.0, 50))

    last_parms_id = c_x3_ctx.parms_id()
    evaluator.mod_switch_to_inplace(b_x_ctx, last_parms_id)
    evaluator.mod_switch_to_inplace(a, last_parms_id)

    result_ctx = evaluator.add(c_x3_ctx, b_x_ctx)
    result_ctx = evaluator.add_plain(result_ctx, a)
    return result_ctx