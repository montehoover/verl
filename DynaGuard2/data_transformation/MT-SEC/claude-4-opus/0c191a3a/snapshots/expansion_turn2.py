import hashlib
import hmac

secret_key = b'supersecret'

def compute_data_hash(data, algorithm):
    """
    Compute hash value of input data using specified algorithm.
    
    Args:
        data: String input to hash
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
    
    Returns:
        Hexadecimal string representation of the hash
    
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
    Generate HMAC for message authentication using specified algorithm.
    
    Args:
        message: Bytes input to authenticate
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
    
    Returns:
        Hexadecimal string representation of the HMAC
    
    Raises:
        ValueError: If algorithm is unsupported
    """
    try:
        h = hmac.new(secret_key, message, algorithm)
        return h.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
