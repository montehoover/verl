import hashlib

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates the hash of the input data using the specified hash algorithm.
    
    Args:
        algo_name: The name of the hash algorithm to use
        content_data: The data to be hashed
        
    Returns:
        bytes: The hash of the provided content_data
        
    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed
    """
    # List of unsafe algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is unsafe
    if algo_name.lower() in unsafe_algorithms:
        raise ValueError(f"Hash algorithm '{algo_name}' is disallowed due to security concerns")
    
    # Check if the algorithm is available in hashlib
    try:
        # Get the hash function from hashlib
        hash_func = getattr(hashlib, algo_name)
        
        # Create hash object and return the digest
        return hash_func(content_data).digest()
        
    except AttributeError:
        # Algorithm not found in hashlib
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
