import hashlib

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return in hexadecimal format."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_flexible_hash(input_string, algorithm):
    """Generate hash of input string using specified algorithm and return in hexadecimal format."""
    if algorithm.lower() == 'sha256':
        hash_object = hashlib.sha256(input_string.encode())
    elif algorithm.lower() == 'sha512':
        hash_object = hashlib.sha512(input_string.encode())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return hash_object.hexdigest()
