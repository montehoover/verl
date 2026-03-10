import hashlib

def encode_password(password):
    """
    Convert a password string to its byte representation.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The password encoded as bytes
    """
    return password.encode('utf-8')

def compute_hash(encoded_password, algorithm):
    """
    Compute the hash value of an encoded password using the specified algorithm.
    
    Args:
        encoded_password (bytes): The byte-encoded password
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
        
    Returns:
        str: The hexadecimal hash value
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(encoded_password)
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def secure_hash(algo_name, pw):
    """
    Generate a cryptographic hash for a given password using the specified algorithm.
    
    Args:
        algo_name (str): The hash algorithm to use
        pw (str): The password to hash
        
    Returns:
        bytes: The hash value as bytes
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hash_obj = hashlib.new(algo_name)
        hash_obj.update(pw.encode('utf-8'))
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
