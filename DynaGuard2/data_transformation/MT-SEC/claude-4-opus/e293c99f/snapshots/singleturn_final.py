import hashlib
import hmac

top_secret_key = b'supersecret'

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a specific cryptographic hash function.
    
    Args:
        algo_name: str, the name of the hashing algorithm to use
        raw_data: bytes, the raw data that needs to be signed
        
    Returns:
        bytes, the generated signature for the provided data
        
    Raises:
        ValueError: if an unsupported or insecure hash algorithm is specified
    """
    # Define secure algorithms (exclude MD5, SHA1, and MD5-SHA1 as they are considered insecure)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s', 'sm3'
    }
    
    # Check if the algorithm is secure
    if algo_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")
    
    # Get the hash algorithm from hashlib
    try:
        hash_algo = getattr(hashlib, algo_name)
    except AttributeError:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")
    
    # Create HMAC signature
    return hmac.new(top_secret_key, raw_data, hash_algo).digest()
