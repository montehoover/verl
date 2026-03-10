import hashlib

def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algorithm_name: str, the name of the cryptographic algorithm to utilize for hashing
        user_password: bytes, this represents the password to be hashed
        
    Returns:
        bytes: the computed hash value of the specified password
        
    Raises:
        ValueError: if the given hash algorithm is not found in the list of supported algorithms
    """
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(user_password)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
