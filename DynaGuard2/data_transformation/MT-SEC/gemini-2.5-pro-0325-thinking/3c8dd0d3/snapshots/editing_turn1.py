import hashlib

def hash_password(pwd: str) -> str:
    """
    Hashes a password using SHA-256.

    Args:
        pwd: The password string to hash.

    Returns:
        The hexadecimal representation of the SHA-256 hash.
    """
    # Encode the password to bytes, as hashlib operates on bytes
    password_bytes = pwd.encode('utf-8')
    
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the password bytes
    sha256_hash.update(password_bytes)
    
    # Get the hexadecimal representation of the hash
    hashed_password = sha256_hash.hexdigest()
    
    return hashed_password
