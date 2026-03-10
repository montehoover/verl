import hashlib

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Apply a specified cryptographic hash to a password.
    
    Args:
        algorithm_name: The name of the cryptographic algorithm to be used
        pwd: The password to hash
        
    Returns:
        The resulting hash value of the password as bytes
        
    Raises:
        ValueError: If the chosen hash algorithm is not available
    """
    # Check if the algorithm is available in hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
    
    # Get the hash function
    hash_func = hashlib.new(algorithm_name)
    
    # Update with the password (convert to bytes if string)
    if isinstance(pwd, str):
        hash_func.update(pwd.encode('utf-8'))
    else:
        hash_func.update(pwd)
    
    # Return the digest (hash value as bytes)
    return hash_func.digest()
