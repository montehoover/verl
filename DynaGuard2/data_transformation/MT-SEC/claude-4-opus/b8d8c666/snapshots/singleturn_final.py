import hashlib
import hmac

secret_key = b'supersecret'

def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.
    
    Args:
        algorithm_name: str - Name of the hash algorithm to use
        payload: bytes - The raw input data to be signed
        
    Returns:
        bytes - A cryptographic signature computed from the given data
        
    Raises:
        ValueError - If the hash algorithm provided is unsupported or disallowed
    """
    # Define secure hash algorithms
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s', 'sm3'
    }
    
    # Check if the algorithm is secure and supported
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Get the hash algorithm from hashlib
    try:
        hash_algorithm = getattr(hashlib, algorithm_name)
    except AttributeError:
        raise ValueError(f"Hash algorithm '{algorithm_name}' not found in hashlib")
    
    # Create and return the HMAC signature
    return hmac.new(secret_key, payload, hash_algorithm).digest()
