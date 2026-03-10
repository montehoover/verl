import hashlib
import hmac

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

def create_hmac(algorithm_name, message, secret_key):
    """
    Generate an HMAC for a given message using a secret key and specified hash algorithm.
    
    Args:
        algorithm_name (str): The name of the hash algorithm to use
        message (bytes): The message to authenticate
        secret_key (bytes): The secret key for HMAC generation
        
    Returns:
        bytes: The HMAC value
        
    Raises:
        ValueError: If the hash algorithm is unsupported
    """
    try:
        return hmac.new(secret_key, message, algorithm_name).digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
