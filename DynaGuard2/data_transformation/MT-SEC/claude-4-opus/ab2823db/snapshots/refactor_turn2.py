import hashlib


def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a password using the specified algorithm.
    
    This function takes a plaintext password and applies the requested
    cryptographic hash function to it, returning the result as a
    hexadecimal string.
    
    Args:
        algo_name (str): The name of the hash algorithm to use.
                        Must be one of the algorithms supported by hashlib.
        secret (str): The plaintext password to be hashed.
        
    Returns:
        str: The hexadecimal representation of the hashed password.
        
    Raises:
        ValueError: If the specified hash algorithm is not supported
                   or unavailable in the hashlib module.
                   
    Example:
        >>> hash_password('sha256', 'my_secure_password')
        'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3'
    """
    try:
        # Create a new hash object using the specified algorithm
        hash_obj = hashlib.new(algo_name)
        
        # Convert the password string to bytes using UTF-8 encoding
        # and update the hash object with these bytes
        hash_obj.update(secret.encode('utf-8'))
        
        # Generate and return the hexadecimal digest of the hash
        return hash_obj.hexdigest()
        
    except ValueError:
        # If hashlib doesn't recognize the algorithm name,
        # raise a more descriptive error message
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
