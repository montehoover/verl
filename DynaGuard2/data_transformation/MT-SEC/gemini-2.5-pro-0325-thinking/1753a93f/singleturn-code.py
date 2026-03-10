import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hashes a password using the chosen algorithm and returns its hexadecimal representation.

    Args:
        hash_algo: The hash function to apply (e.g., 'sha256', 'md5').
        pwd: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    if hash_algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}")

    try:
        # Create a new hash object
        hasher = hashlib.new(hash_algo)
        
        # Encode the password string to bytes (UTF-8 is a common choice)
        password_bytes = pwd.encode('utf-8')
        
        # Update the hash object with the password bytes
        hasher.update(password_bytes)
        
        # Get the hexadecimal representation of the hash
        hex_digest = hasher.hexdigest()
        
        return hex_digest
    except Exception as e:
        # Catch any other hashlib related errors, though `new` should handle algorithm availability.
        # This is more of a safeguard.
        raise ValueError(f"Error during hashing with {hash_algo}: {e}")
