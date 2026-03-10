import hashlib

def basic_hash(data, algorithm):
    """
    Compute hash value for given data using specified algorithm.
    
    Args:
        data: byte sequence to hash
        algorithm: hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        Hash value as hexadecimal string
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()

def is_algorithm_secure(algorithm):
    """
    Check if a hash algorithm is considered secure.
    
    Args:
        algorithm: hash algorithm name
    
    Returns:
        bool: True if algorithm is secure, False otherwise
    """
    insecure_algorithms = {'md5', 'sha1'}
    return algorithm.lower() not in insecure_algorithms
