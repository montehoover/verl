import hashlib

def encode_password(password):
    """
    Convert a plaintext password to its byte representation.
    
    Args:
        password (str): The plaintext password to encode
        
    Returns:
        bytes: The UTF-8 encoded byte representation of the password
    """
    return password.encode('utf-8')

def select_and_apply_hash(encoded_password, algorithm_name):
    """
    Apply a specified hash algorithm to an encoded password.
    
    Args:
        encoded_password (bytes): The byte-encoded password to hash
        algorithm_name (str): The name of the hash algorithm to use
        
    Returns:
        str: The hexadecimal representation of the hashed password
        
    Raises:
        ValueError: If the specified algorithm is not supported
    """
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(encoded_password)
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
