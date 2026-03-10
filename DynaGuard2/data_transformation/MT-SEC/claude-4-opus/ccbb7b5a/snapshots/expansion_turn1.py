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
