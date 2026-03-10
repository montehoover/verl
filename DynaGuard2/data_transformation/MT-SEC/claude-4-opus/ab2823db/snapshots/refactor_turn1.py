import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a password using the specified algorithm.
    
    Args:
        algo_name: The hash function to apply
        secret: The plaintext password to be hashed
        
    Returns:
        A string containing the hexadecimal representation of the hashed password
        
    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided
    """
    try:
        # Create a hash object for the specified algorithm
        hash_obj = hashlib.new(algo_name)
        # Update the hash object with the password encoded as bytes
        hash_obj.update(secret.encode('utf-8'))
        # Return the hexadecimal representation
        return hash_obj.hexdigest()
    except ValueError:
        # Raise ValueError if the algorithm is not supported
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
