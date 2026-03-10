import hashlib

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Applies a specified cryptographic hash to a password.
    
    Args:
        algorithm_name: The name of the cryptographic algorithm to be used
        pwd: The password to hash
        
    Returns:
        The resulting hash value of the password as bytes
        
    Raises:
        ValueError: If the chosen hash algorithm is not available
    """
    # Get list of available algorithms
    available_algorithms = hashlib.algorithms_available
    
    # Check if the requested algorithm is available
    if algorithm_name not in available_algorithms:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
    
    # Create hash object and compute hash
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(pwd.encode('utf-8'))
    
    return hash_obj.digest()
