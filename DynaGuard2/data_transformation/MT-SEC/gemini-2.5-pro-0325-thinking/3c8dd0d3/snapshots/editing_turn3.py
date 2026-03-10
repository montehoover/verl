import hashlib

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Applies a cryptographic hash to a password using the specified algorithm.

    Args:
        algorithm_name: The name of the cryptographic hash algorithm 
                        (e.g., 'sha256', 'sha512').
        pwd: The password string to hash.

    Returns:
        The resulting hash value as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not available.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Encode the password to bytes, as hashlib operates on bytes
    password_bytes = pwd.encode('utf-8')
    
    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(algorithm_name)
    
    # Update the hash object with the password bytes
    hash_object.update(password_bytes)
    
    # Get the hash value as bytes
    hashed_password_bytes = hash_object.digest()
    
    return hashed_password_bytes
