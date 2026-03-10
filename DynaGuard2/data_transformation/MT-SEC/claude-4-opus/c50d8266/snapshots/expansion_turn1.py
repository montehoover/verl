import hashlib

def basic_hash(data, algorithm):
    """
    Generate a hash of the input data using the specified algorithm.
    
    Args:
        data (str): The input string to hash
        algorithm (str): The hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal hash value
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # Get list of available algorithms
    available_algorithms = hashlib.algorithms_available
    
    # Check if the algorithm is supported
    if algorithm not in available_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Available algorithms: {', '.join(sorted(available_algorithms))}")
    
    # Create hash object
    hash_obj = hashlib.new(algorithm)
    
    # Update hash with data (convert string to bytes)
    hash_obj.update(data.encode('utf-8'))
    
    # Return hexadecimal digest
    return hash_obj.hexdigest()
