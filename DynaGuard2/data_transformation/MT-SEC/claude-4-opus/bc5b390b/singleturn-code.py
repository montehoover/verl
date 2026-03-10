import hashlib

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates the hash of the input data using the specified hash algorithm.
    
    Args:
        algorithm_name: The name of the hash algorithm to use
        content: The input data to be hashed
        
    Returns:
        The hash of the provided content as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed
    """
    # Define unsafe algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is unsafe
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is disallowed due to security concerns")
    
    # Check if the algorithm is available in hashlib
    try:
        # Try to get the hash function
        hash_func = hashlib.new(algorithm_name)
    except ValueError:
        # Algorithm is not available
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
    
    # Update the hash with the content and return the digest
    hash_func.update(content)
    return hash_func.digest()
