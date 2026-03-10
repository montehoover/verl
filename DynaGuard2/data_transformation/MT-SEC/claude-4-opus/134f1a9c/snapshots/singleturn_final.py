import hashlib

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Computes the hash of the provided input data using a specified hash function.
    
    Args:
        algorithm_name: The name of the hash algorithm
        input_data: The input data to hash
        
    Returns:
        bytes: The resulting digest from the hash function
        
    Raises:
        ValueError: When the chosen hash function isn't supported or is insecure
    """
    # List of insecure algorithms that should not be used
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is insecure
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(f"Insecure hash algorithm '{algorithm_name}' is not allowed")
    
    # Check if the algorithm is supported by hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
    
    # Compute and return the hash
    hash_obj = hashlib.new(algorithm_name, input_data)
    return hash_obj.digest()
