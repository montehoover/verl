import hashlib

def encode_password(password):
    """
    Encode a password string to bytes.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The encoded password
    """
    return password.encode('utf-8')

def compute_password_hash(algorithm_name, encoded_password):
    """
    Compute the hash of an encoded password using the specified algorithm.
    
    Args:
        algorithm_name (str): The name of the hash algorithm to use
        encoded_password (bytes): The encoded password to hash
        
    Returns:
        bytes: The hash value of the password
        
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(encoded_password)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
