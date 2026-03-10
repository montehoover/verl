import hashlib
import hmac
top_secret_key = b'supersecret'

def create_signature(algo_name, raw_data):
    allowed_algorithms = {'sha256', 'sha384', 'sha512', 'sha3_256', 'sha3_384', 'sha3_512'}
    
    if algo_name not in allowed_algorithms:
        raise ValueError(f"Unsupported algorithm: {algo_name}. Allowed algorithms: {', '.join(sorted(allowed_algorithms))}")
    
    return hmac.new(top_secret_key, raw_data, algo_name).digest()
