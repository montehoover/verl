import hashlib

def hash_data(data, algorithm):
    """
    Hash the input data using the specified algorithm.
    
    Args:
        data (str): The input string to hash
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal hash value of the input data
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
