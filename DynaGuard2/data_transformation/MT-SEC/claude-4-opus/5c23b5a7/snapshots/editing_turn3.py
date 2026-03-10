import hashlib

def secure_hash_password(algorithm_name, user_password):
    # Check if the algorithm is supported
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not supported")
    
    # Get the hash algorithm from hashlib
    hash_algorithm = getattr(hashlib, algorithm_name)
    
    # Create hash of the password
    hash_object = hash_algorithm(user_password)
    
    return hash_object.digest()
