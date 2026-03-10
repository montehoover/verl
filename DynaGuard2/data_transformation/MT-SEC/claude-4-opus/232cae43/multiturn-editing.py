import hashlib
import hmac

def generate_hash(input_string):
    """Generate SHA-256 hash of the input string."""
    return hashlib.sha256(input_string.encode()).hexdigest()

def generate_hmac(input_data, secret_key):
    """Generate HMAC-SHA256 signature of the input data using the secret key."""
    return hmac.new(secret_key, input_data, hashlib.sha256).hexdigest()

def compute_hmac(hash_algorithm, payload, secret):
    """Compute HMAC signature using the specified hash algorithm.
    
    Args:
        hash_algorithm: String specifying the hash function to use
        payload: The input data as bytes
        secret: The secret key as bytes
        
    Returns:
        The HMAC signature as a hexadecimal string
        
    Raises:
        ValueError: If the hash algorithm is not supported
    """
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    hash_func = getattr(hashlib, hash_algorithm)
    return hmac.new(secret, payload, hash_func).hexdigest()
