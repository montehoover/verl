import hashlib

def derive_hash(algorithm_name, input_data):
    insecure_algorithms = {'md5', 'sha1'}
    
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm '{algorithm_name}' is not allowed")
    
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(input_data)
    return hash_obj.digest()
