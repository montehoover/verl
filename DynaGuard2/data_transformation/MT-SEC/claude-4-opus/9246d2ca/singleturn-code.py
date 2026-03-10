import hashlib

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algo_name: str - The name of the cryptographic algorithm to utilize for hashing
        pass_key: bytes - The password to be hashed
        
    Returns:
        bytes - The computed hash value of the specified password
        
    Raises:
        ValueError - If the given hash algorithm is not found in the list of supported algorithms
    """
    try:
        # Get the hash algorithm from hashlib
        hash_func = hashlib.new(algo_name)
        # Update with the password
        hash_func.update(pass_key)
        # Return the digest
        return hash_func.digest()
    except ValueError:
        # If the algorithm is not supported, hashlib.new() raises ValueError
        raise ValueError(f"Hash algorithm '{algo_name}' is not supported")
