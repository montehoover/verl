import hashlib

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hexadecimal representation.
    
    Args:
        algorithm_name: The hash algorithm to be used for hashing
        raw_password: The input password to be hashed
        
    Returns:
        A string containing the password's hashed hexadecimal value
        
    Raises:
        ValueError: When an unsupported hash function is chosen
    """
    # Get list of available algorithms
    available_algorithms = hashlib.algorithms_available
    
    # Check if the requested algorithm is available
    if algorithm_name not in available_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. Available algorithms: {sorted(available_algorithms)}")
    
    # Get the hash function
    hash_func = hashlib.new(algorithm_name)
    
    # Update with the password (convert to bytes if needed)
    if isinstance(raw_password, str):
        hash_func.update(raw_password.encode('utf-8'))
    else:
        hash_func.update(raw_password)
    
    # Return the hexadecimal digest
    return hash_func.hexdigest()
