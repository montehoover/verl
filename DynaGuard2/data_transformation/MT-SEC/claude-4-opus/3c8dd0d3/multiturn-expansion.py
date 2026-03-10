import hashlib

def encode_password(password):
    """
    Encode a password string to bytes for secure storage.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The password encoded as bytes
    """
    return password.encode('utf-8')

def compute_password_hash(encoded_password, algorithm):
    """
    Compute the hash value of an encoded password.
    
    Args:
        encoded_password (bytes): The byte-encoded password
        algorithm (str): The hash algorithm name (e.g., 'sha256', 'md5')
        
    Returns:
        str: The hexadecimal hash value of the password
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(encoded_password)
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def encrypt_password(algorithm_name, pwd):
    """
    Encrypt a password using the specified hash algorithm.
    
    Args:
        algorithm_name (str): The name of the hash algorithm to use
        pwd (str): The password to encrypt
        
    Returns:
        bytes: The hashed password as bytes
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(pwd.encode('utf-8'))
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
