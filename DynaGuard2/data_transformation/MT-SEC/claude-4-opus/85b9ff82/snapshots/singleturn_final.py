import hashlib
import hmac
secret_key = b'supersecret'

def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.
    
    Args:
        hash_algorithm: Name of the hash algorithm to use
        content: Raw input data to be signed
        
    Returns:
        Cryptographic signature computed from the given data
        
    Raises:
        ValueError: If the hash algorithm is unsupported or disallowed
    """
    # Define secure hash algorithms (excluding weak ones like MD5, SHA1, etc.)
    SECURE_ALGORITHMS = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is secure and supported
    if hash_algorithm not in SECURE_ALGORITHMS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {hash_algorithm}")
    
    # Get the hash function from hashlib
    try:
        hash_func = getattr(hashlib, hash_algorithm)
    except AttributeError:
        raise ValueError(f"Hash algorithm not found: {hash_algorithm}")
    
    # Create and return the HMAC signature
    return hmac.new(secret_key, content, hash_func).digest()
