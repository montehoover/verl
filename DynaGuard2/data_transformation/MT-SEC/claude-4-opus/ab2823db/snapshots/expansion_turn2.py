import hashlib

def encode_password(password):
    """
    Takes a password as a string and returns its byte representation.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The UTF-8 encoded byte representation of the password
    """
    return password.encode('utf-8')

def compute_hash(encoded_password, algorithm):
    """
    Computes the hash value for an encoded password using the specified algorithm.
    
    Args:
        encoded_password (bytes): The byte-encoded password
        algorithm (str): The hash algorithm name (e.g., 'sha256', 'md5')
        
    Returns:
        str: The hash value in hexadecimal format
        
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(encoded_password)
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
