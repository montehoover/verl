import hashlib
import hmac

def compute_basic_hash(data, algorithm):
    """
    Compute hash value of input data using specified algorithm.
    
    Args:
        data: String input to hash
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5')
    
    Returns:
        Hexadecimal string representation of the hash
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # Check if algorithm is supported
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Create hash object
    hash_obj = hashlib.new(algorithm)
    
    # Update hash with encoded data
    hash_obj.update(data.encode('utf-8'))
    
    # Return hexadecimal digest
    return hash_obj.hexdigest()


def generate_hmac_with_key(message, algorithm, secret_key):
    """
    Generate HMAC for message authentication using a secret key.
    
    Args:
        message: Bytes message to authenticate
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5')
        secret_key: Secret key for HMAC generation
    
    Returns:
        Hexadecimal string representation of the HMAC
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    # Check if algorithm is supported
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Create HMAC object
    hmac_obj = hmac.new(secret_key, message, algorithm)
    
    # Return hexadecimal digest
    return hmac_obj.hexdigest()
