import hashlib
import hmac

secret_key = b'supersecret'

def compute_basic_hash(data, algorithm):
    """
    Compute the hash of input data using the specified algorithm.
    
    Args:
        data: String input to be hashed
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        The hexadecimal hash value of the input data
    
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
    Generate HMAC for message authentication using the specified algorithm.
    
    Args:
        message: Bytes input to be authenticated
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        The hexadecimal HMAC value of the message
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        # Verify the algorithm is supported
        hashlib.new(algorithm)
        # Generate HMAC
        hmac_obj = hmac.new(secret_key, message, algorithm)
        return hmac_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
