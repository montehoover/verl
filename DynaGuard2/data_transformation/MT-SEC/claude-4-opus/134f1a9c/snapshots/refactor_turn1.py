import hashlib

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    # List of insecure hash algorithms that should not be used
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is insecure
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm '{algorithm_name}' is not supported")
    
    # Check if the algorithm is available in hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
    
    # Create hash object and compute digest
    hash_obj = hashlib.new(algorithm_name, input_data)
    return hash_obj.digest()
