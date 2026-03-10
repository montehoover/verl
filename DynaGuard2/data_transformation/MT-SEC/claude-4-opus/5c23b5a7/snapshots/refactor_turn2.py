import hashlib


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    This function takes a password in bytes format and hashes it using the specified
    cryptographic algorithm from Python's hashlib module.
    
    Args:
        algorithm_name: str
            The name of the cryptographic algorithm to utilize for hashing.
            Must be one of the algorithms supported by hashlib (e.g., 'sha256', 
            'sha512', 'md5', 'blake2b', etc.).
        user_password: bytes
            The password to be hashed, provided as a bytes object.
        
    Returns:
        bytes: The computed hash value of the specified password.
        
    Raises:
        ValueError: If the given hash algorithm is not found in the list of 
                   supported algorithms.
                   
    Example:
        >>> password = b"my_secure_password"
        >>> hashed = secure_hash_password("sha256", password)
        >>> len(hashed)
        32
    """
    try:
        # Create a new hash object using the specified algorithm
        hash_obj = hashlib.new(algorithm_name)
        
        # Update the hash object with the password bytes
        hash_obj.update(user_password)
        
        # Return the computed hash digest as bytes
        return hash_obj.digest()
        
    except ValueError:
        # Re-raise with a more descriptive error message
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
