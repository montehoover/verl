import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hashes a password using the specified hash algorithm.
    
    Args:
        hash_algo: The hash function to apply
        pwd: The plaintext password to be hashed
        
    Returns:
        A string containing the hexadecimal representation of the hashed password
        
    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided
    """
    try:
        # Create a hash object for the specified algorithm
        hash_obj = hashlib.new(hash_algo)
        # Update the hash object with the password bytes
        hash_obj.update(pwd.encode('utf-8'))
        # Return the hexadecimal representation
        return hash_obj.hexdigest()
    except ValueError:
        # Re-raise with a more descriptive message
        raise ValueError(f"Unsupported or unavailable hash algorithm: '{hash_algo}'")
