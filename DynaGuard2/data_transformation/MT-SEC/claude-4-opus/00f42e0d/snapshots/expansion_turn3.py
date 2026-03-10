import hashlib
import hmac

key = b"supersecret"

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

def create_hmac(message, hash_name):
    """
    Create an HMAC for a message using the specified hash algorithm.
    
    Args:
        message (bytes): The message to authenticate
        hash_name (str): The hash algorithm name (e.g., 'sha256', 'sha512')
    
    Returns:
        hmac.HMAC: The HMAC object
    
    Raises:
        ValueError: If the hash algorithm is not supported
    """
    try:
        return hmac.new(key, message, hash_name)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")

def sign_message(hash_name, message, key):
    """
    Sign a message using HMAC with the specified hash algorithm.
    
    Args:
        hash_name (str): The hash algorithm name (e.g., 'sha256', 'sha512')
        message (bytes): The message to sign
        key (bytes): The secret key for signing
    
    Returns:
        bytes: The signed message (HMAC digest)
    
    Raises:
        ValueError: If the hash algorithm is not supported
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")
    
    try:
        h = hmac.new(key, message, hash_name)
        return h.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")
