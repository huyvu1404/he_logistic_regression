from seal import EncryptionParameters, scheme_type, CoeffModulus, SEALContext, KeyGenerator, Encryptor, Evaluator, Decryptor, CKKSEncoder # type: ignore
from utils import print_parameters

def setup_ckks_params(poly_modulus_degree, coef_modulus_chain):
    
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, coef_modulus_chain))
    context = SEALContext(parms)
    
    print_parameters(context)
    return context

def create_tools(context):
   
    keygen = KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()
    galois_keys = keygen.create_galois_keys()
    
    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)
    ckks_encoder = CKKSEncoder(context)
    return secret_key, public_key, relin_keys, galois_keys, encryptor, decryptor, evaluator, ckks_encoder
