import hashlib

def compute_message_hash(message, algorithm):
    """
    Compute the hash of a message using the specified algorithm.
    
    Args:
        message (str): The message to hash
        algorithm (str): The hash algorithm name (e.g., 'sha256', 'sha512')
    
    Returns:
        bytes: The computed hash as bytes
    
    Raises:
        ValueError: If the algorithm is not supported
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(message.encode('utf-8'))
    return hash_obj.digest()
