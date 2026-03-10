import hashlib
top_secret_key = b'supersecret'

def create_signature(algo_name, raw_data):
    allowed_algorithms = {'sha256', 'sha384', 'sha512', 'sha3_256', 'sha3_384', 'sha3_512'}
    
    if algo_name not in allowed_algorithms:
        raise ValueError(f"Unsupported algorithm: {algo_name}. Allowed algorithms: {', '.join(sorted(allowed_algorithms))}")
    
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(raw_data)
    return hash_obj.digest()
