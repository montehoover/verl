import hashlib

def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algorithm_name: str - The name of the cryptographic algorithm to utilize for hashing
        user_password: bytes - The password to be hashed
    
    Returns:
        bytes - The computed hash value of the specified password
    
    Raises:
        ValueError - If the given hash algorithm is not found in the list of supported algorithms
    """
    # Get the list of available algorithms
    available_algorithms = hashlib.algorithms_available
    
    # Check if the algorithm is supported
    if algorithm_name not in available_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is not supported. Available algorithms: {available_algorithms}")
    
    # Create a hash object using the specified algorithm
    hash_obj = hashlib.new(algorithm_name, user_password)
    
    # Return the computed hash as bytes
    return hash_obj.digest()
