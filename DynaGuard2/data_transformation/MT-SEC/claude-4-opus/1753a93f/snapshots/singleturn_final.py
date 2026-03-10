import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a password using the specified algorithm and return its hexadecimal representation.
    
    Args:
        hash_algo: str, the hash function to apply
        pwd: str, the plaintext password to be hashed
        
    Returns:
        str, a string containing the hexadecimal representation of the hashed password
        
    Raises:
        ValueError: if an unsupported or unavailable hash algorithm is provided
    """
    try:
        # Get the hash function from hashlib
        hash_func = getattr(hashlib, hash_algo)
        
        # Create hash object and update with password bytes
        hash_obj = hash_func()
        hash_obj.update(pwd.encode('utf-8'))
        
        # Return hexadecimal representation
        return hash_obj.hexdigest()
        
    except AttributeError:
        # If the algorithm doesn't exist in hashlib, raise ValueError
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}")
