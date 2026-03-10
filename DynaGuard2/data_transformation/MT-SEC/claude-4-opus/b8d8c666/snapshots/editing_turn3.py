import hashlib
import hmac

secret_key = b'supersecret'

def generate_hash(input_string):
    """Generate SHA-256 hash of input string and return hexadecimal format."""
    return hashlib.sha256(input_string.encode()).hexdigest()

def generate_hmac(input_string, secret_key):
    """Generate HMAC-SHA256 signature of input string using secret key."""
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).hexdigest()

def create_signature(algorithm_name, payload):
    """Generate a cryptographic signature using a specified hash algorithm.
    
    Args:
        algorithm_name: A string specifying the hash algorithm
        payload: The raw data as bytes
        
    Returns:
        A cryptographic signature using the specified secure algorithm
        
    Raises:
        ValueError: For unsupported or disallowed algorithms
    """
    supported_algorithms = {'sha512_256', 'sha224', 'sha384', 'sha512', 'sm3', 'sha1', 
                           'sha3_384', 'shake_128', 'sha256', 'blake2b', 'md5-sha1', 
                           'shake_256', 'ripemd160', 'md5', 'sha3_256', 'blake2s', 
                           'sha3_512', 'sha3_224'}
    
    if algorithm_name not in supported_algorithms:
        raise ValueError(f"Unsupported or disallowed algorithm: {algorithm_name}")
    
    hash_func = getattr(hashlib, algorithm_name)
    return hmac.new(secret_key, payload, hash_func).hexdigest()
