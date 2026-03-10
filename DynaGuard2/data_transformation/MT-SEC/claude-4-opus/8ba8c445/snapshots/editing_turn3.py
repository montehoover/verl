import hashlib

def hash_data(algo_name, content_data):
    # Check if the algorithm is available in hashlib
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algo_name}' is unavailable or disallowed")
    
    # Create hash object with the specified algorithm
    hash_object = hashlib.new(algo_name, content_data)
    
    # Return the hash digest
    return hash_object.digest()
