import hashlib
import hmac

secret_key = b'supersecret'

def compute_hash(data, algorithm):
    """
    Compute hash value of input data using specified algorithm.
    
    Args:
        data: String input to hash
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
        
    Returns:
        Hexadecimal hash value as string
        
    Raises:
        ValueError: If algorithm is unsupported
    """
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def generate_hmac(message, algorithm):
    """
    Generate HMAC for message authentication.
    
    Args:
        message: Bytes message to authenticate
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
        
    Returns:
        Hexadecimal HMAC value as string
        
    Raises:
        ValueError: If algorithm is unsupported
    """
    try:
        hmac_obj = hmac.new(secret_key, message, algorithm)
        return hmac_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
