import hashlib
import hmac

secret_key = b'supersecret'

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return hexadecimal format."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()

def generate_hmac_signature(input_string, secret_key):
    """Generate HMAC signature using SHA-256.
    
    Args:
        input_string: The message data to sign
        secret_key: The secret key as bytes
        
    Returns:
        bytes: The HMAC signature
    """
    return hmac.new(secret_key, input_string.encode(), hashlib.sha256).digest()

def create_signature(algorithm_name, message):
    """Generate a cryptographic signature for the provided data using a specified hash function.
    
    Args:
        algorithm_name: A string of the hash algorithm to use
        message: The raw data as bytes
        
    Returns:
        bytes: The HMAC signature using the specified algorithm
        
    Raises:
        ValueError: For unsupported or insecure algorithms
    """
    # List of insecure algorithms
    insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if algorithm is insecure
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(f"Insecure algorithm '{algorithm_name}' is not supported")
    
    # Check if algorithm is supported by hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported algorithm '{algorithm_name}'")
    
    # Get the hash function
    hash_func = getattr(hashlib, algorithm_name)
    
    # Generate and return HMAC signature
    return hmac.new(secret_key, message, hash_func).digest()
