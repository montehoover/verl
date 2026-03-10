import hashlib

def encode_password(password):
    """
    Encode a password string to bytes.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The password encoded as bytes
    """
    return password.encode('utf-8')

def compute_hash(hash_name, data):
    """
    Compute the hash value of data using the specified hash algorithm.
    
    Args:
        hash_name (str): The name of the hash algorithm to use
        data (bytes): The data to hash
        
    Returns:
        bytes: The hash value of the data
        
    Raises:
        ValueError: If the hash algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(hash_name)
        hash_obj.update(data)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")

def hash_password(hash_name, password):
    """
    Hash a password using the specified hash algorithm.
    
    Args:
        hash_name (str): The name of the hash algorithm to use
        password (str): The password to hash
        
    Returns:
        bytes: The hash value of the password
        
    Raises:
        ValueError: If the hash algorithm is not available
    """
    encoded_password = encode_password(password)
    return compute_hash(hash_name, encoded_password)
