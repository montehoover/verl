import hashlib

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hexadecimal representation.
    
    Args:
        algorithm_name: The hash algorithm to be used for hashing
        raw_password: The input password to be hashed
        
    Returns:
        A string containing the password's hashed hexadecimal value
        
    Raises:
        ValueError: When an unsupported hash function is chosen
    """
    try:
        # Create a hash object with the specified algorithm
        hash_obj = hashlib.new(algorithm_name)
        # Update the hash object with the password bytes
        hash_obj.update(raw_password.encode('utf-8'))
        # Return the hexadecimal representation
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
