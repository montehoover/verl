import hashlib

def secure_hash_password(algorithm_name: str, user_password: bytes) -> str:
    """
    Hashes a password using a specified secure algorithm and returns the hex digest.
    Raises ValueError if the algorithm is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hashing algorithm: {algorithm_name}")
    
    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(algorithm_name)
    
    # Update the hash object with the user_password (already bytes)
    hash_object.update(user_password)
    
    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()
    
    return hex_digest
