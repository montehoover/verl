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

def produce_signature(method_name, data, secret_key):
    """
    Sign a message using HMAC with the specified hash algorithm.
    
    Args:
        method_name: Hash algorithm name ('sha256', 'md5', etc.)
        data: Bytes message to be signed
        secret_key: Bytes secret key for HMAC generation
    
    Returns:
        HMAC signature as bytes
    
    Raises:
        ValueError: If the specified hash algorithm is not supported
    """
    try:
        # Create HMAC object with the secret key and specified algorithm
        hmac_obj = hmac.new(secret_key, data, getattr(hashlib, method_name))
        # Return the HMAC digest as bytes
        return hmac_obj.digest()
    except AttributeError:
        raise ValueError(f"Unsupported hash algorithm: {method_name}")
