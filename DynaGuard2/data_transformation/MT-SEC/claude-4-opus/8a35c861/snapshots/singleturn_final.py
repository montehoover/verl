import hashlib

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Apply a cryptographic hash to a password using the specified algorithm.
    
    Args:
        algo_name: str - The name of the cryptographic algorithm to be used
        pw: str - The password to hash
        
    Returns:
        bytes - The resulting hash value of the password
        
    Raises:
        ValueError - If the chosen hash algorithm is not available
    """
    try:
        # Get the hash algorithm from hashlib
        hash_func = hashlib.new(algo_name)
        # Update the hash with the password (convert string to bytes)
        hash_func.update(pw.encode('utf-8'))
        # Return the digest (hash value) as bytes
        return hash_func.digest()
    except ValueError:
        # Re-raise ValueError if the algorithm is not available
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
