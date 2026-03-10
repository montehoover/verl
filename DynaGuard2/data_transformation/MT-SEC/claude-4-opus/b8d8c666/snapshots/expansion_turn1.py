import hashlib

def compute_basic_hash(data, algorithm):
    """
    Compute the hash of input data using the specified algorithm.
    
    Args:
        data: String input to be hashed
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        The hexadecimal hash value of the input data
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
