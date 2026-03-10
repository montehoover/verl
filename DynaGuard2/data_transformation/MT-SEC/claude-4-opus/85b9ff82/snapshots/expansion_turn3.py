import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data, algorithm):
    """
    Hash the input data using the specified algorithm.
    
    Args:
        data (str): The input string to hash
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal hash value of the input data
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def generate_hmac(message, algorithm):
    """
    Generate an HMAC for the given message using the specified algorithm.
    
    Args:
        message (bytes): The message to authenticate
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
    
    Returns:
        str: The hexadecimal HMAC value
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hmac_obj = hmac.new(secret_key, message, algorithm)
        return hmac_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def sign_message(hash_algorithm, content):
    """
    Generate a cryptographic signature for the given content using HMAC.
    
    Args:
        hash_algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5')
        content (bytes): The content to sign
    
    Returns:
        bytes: The signature as bytes
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        hmac_obj = hmac.new(secret_key, content, hash_algorithm)
        return hmac_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
