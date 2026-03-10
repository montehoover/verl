import hashlib

def hash_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Hashes a password using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        pwd: The password string to hash.

    Returns:
        The hash value as bytes.
    """
    # Encode the password to bytes, as hashlib operates on bytes
    password_bytes = pwd.encode('utf-8')
    
    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(algorithm_name)
    
    # Update the hash object with the password bytes
    hash_object.update(password_bytes)
    
    # Get the hash value as bytes
    hashed_password_bytes = hash_object.digest()
    
    return hashed_password_bytes
