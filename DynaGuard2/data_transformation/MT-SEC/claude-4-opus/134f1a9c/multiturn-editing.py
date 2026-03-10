import hashlib

def derive_hash(algorithm_name, input_data):
    secure_algorithms = {'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256', 
                        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512', 
                        'blake2b', 'blake2s'}
    
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported hash algorithm '{algorithm_name}'")
    
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(input_data)
    return hash_obj.digest()
