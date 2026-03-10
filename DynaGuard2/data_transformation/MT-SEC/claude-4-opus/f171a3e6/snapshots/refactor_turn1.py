import hashlib

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.
    
    Args:
        hash_name: str, the name of the hash algorithm
        password: str, the password to be hashed
        
    Returns:
        bytes, the hash value of the password
        
    Raises:
        ValueError: when the hash algorithm is not available
    """
    try:
        # Create a hash object with the specified algorithm
        hash_obj = hashlib.new(hash_name)
        # Update the hash object with the password encoded as bytes
        hash_obj.update(password.encode('utf-8'))
        # Return the digest as bytes
        return hash_obj.digest()
    except ValueError:
        # Raise ValueError with a proper message when algorithm is not available
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
