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

def compute_digest(algorithm_name, content):
    """
    Compute hash of given byte sequence using specified algorithm.
    
    Args:
        algorithm_name: Name of hash algorithm
        content: Byte sequence to hash
    
    Returns:
        Hash value as bytes
        
    Raises:
        ValueError: If algorithm is unavailable or unsafe
    """
    if not is_algorithm_secure(algorithm_name):
        raise ValueError(f"Algorithm '{algorithm_name}' is not secure")
    
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(content)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available")
