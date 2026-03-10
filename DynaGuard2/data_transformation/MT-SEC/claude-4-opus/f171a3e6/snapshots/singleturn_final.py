import hashlib

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.
    
    Args:
        hash_name: The name of the hash algorithm
        password: The password to be hashed
        
    Returns:
        The hash value of the password as bytes
        
    Raises:
        ValueError: When the hash algorithm is not available
    """
    try:
        # Get the hash function from hashlib
        hash_func = getattr(hashlib, hash_name)
        # Create hash object, encode password to bytes, and return digest
        return hash_func(password.encode()).digest()
    except AttributeError:
        # If the hash algorithm doesn't exist, raise ValueError
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
