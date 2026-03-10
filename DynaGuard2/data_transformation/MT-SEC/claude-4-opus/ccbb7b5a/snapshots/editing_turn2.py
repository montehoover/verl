import hashlib

def generate_hash(input_string, algorithm='sha256'):
    """Generate hash of input string using specified algorithm and return as hexadecimal string."""
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm}' is not available in hashlib")
    
    hash_object = hashlib.new(algorithm, input_string.encode())
    return hash_object.hexdigest()
