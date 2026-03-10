import hashlib

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algo_name: The name of the cryptographic algorithm to utilize for hashing
        pass_key: The password to be hashed
        
    Returns:
        The computed hash value of the specified password
        
    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms
    """
    try:
        hash_obj = hashlib.new(algo_name)
        hash_obj.update(pass_key)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
