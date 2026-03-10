import hashlib
import hmac
secret_key = b'supersecret'

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generate a cryptographic signature using HMAC with the specified hash algorithm.
    
    Args:
        algorithm: str, the name of the hashing algorithm to use
        message: bytes, the raw data that needs to be signed
    
    Returns:
        bytes, the generated signature for the provided data
    
    Raises:
        ValueError: if an unsupported or insecure hash algorithm is specified
    """
    # Define secure algorithms (excluding MD5, SHA1, and MD5-SHA1 as they are considered insecure)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s', 'sm3'
    }
    
    # Check if the algorithm is supported and secure
    if algorithm not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")
    
    # Get the hash function from hashlib
    try:
        hash_func = getattr(hashlib, algorithm)
    except AttributeError:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")
    
    # Generate and return the HMAC signature
    return hmac.new(secret_key, message, hash_func).digest()
