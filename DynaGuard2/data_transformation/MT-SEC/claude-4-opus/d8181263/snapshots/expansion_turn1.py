import hashlib

def hash_data(data, algorithm):
    """
    Hash the input data using the specified algorithm.
    
    Args:
        data (str): The string data to hash
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal hash value of the input data
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # List of supported algorithms
    supported_algorithms = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512']
    
    # Check if the algorithm is supported
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are: {', '.join(supported_algorithms)}")
    
    # Get the hash function
    hash_func = getattr(hashlib, algorithm)
    
    # Create hash object and update with data
    hash_obj = hash_func()
    hash_obj.update(data.encode('utf-8'))
    
    # Return the hexadecimal digest
    return hash_obj.hexdigest()
