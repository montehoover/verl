import hashlib

def compute_hash(message, algorithm):
    """
    Compute the hash value of a message using the specified algorithm.
    
    Args:
        message (str): The input message to hash
        algorithm (str): The name of the hash algorithm to use
        
    Returns:
        str: The hexadecimal hash value of the message
        
    Raises:
        ValueError: If the hash algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(message.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
