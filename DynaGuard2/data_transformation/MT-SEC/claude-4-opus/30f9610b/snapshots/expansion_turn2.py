import hashlib
import hmac

def compute_basic_hash(data, algorithm):
    """
    Compute the hash value of input data using the specified algorithm.
    
    Args:
        data: String input to be hashed
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
    
    Returns:
        Hexadecimal string representation of the hash
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        # Create hash object for the specified algorithm
        hash_obj = hashlib.new(algorithm)
        # Update hash object with the data (encoded to bytes)
        hash_obj.update(data.encode('utf-8'))
        # Return hexadecimal representation of the hash
        return hash_obj.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def generate_hmac_signature(message, algorithm, secret_key):
    """
    Generate HMAC signature for message authentication.
    
    Args:
        message: Bytes input to be authenticated
        algorithm: Hash algorithm name ('sha256', 'md5', etc.)
        secret_key: Secret key for HMAC generation
    
    Returns:
        Hexadecimal string representation of the HMAC signature
    
    Raises:
        ValueError: If the algorithm is unsupported
    """
    try:
        # Create HMAC object with the secret key and specified algorithm
        hmac_obj = hmac.new(secret_key, message, getattr(hashlib, algorithm))
        # Return hexadecimal representation of the HMAC
        return hmac_obj.hexdigest()
    except AttributeError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
