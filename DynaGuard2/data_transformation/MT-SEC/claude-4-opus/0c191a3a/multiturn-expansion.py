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

def generate_signature(hash_name, data):
    """
    Generate a digital signature for given data using secure hash algorithms.
    
    Args:
        hash_name: Hash algorithm name (str)
        data: Data to sign (bytes)
    
    Returns:
        Signature as bytes
    
    Raises:
        ValueError: If hash algorithm is not available
    """
    try:
        h = hmac.new(secret_key, data, hash_name)
        return h.digest()
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
