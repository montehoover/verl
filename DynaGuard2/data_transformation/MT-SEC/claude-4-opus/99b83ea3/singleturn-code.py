import hashlib
import hmac
secret_key = b'supersecret'

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the provided data using a specified hash function.
    
    Args:
        algorithm_name: str, the desired hash algorithm
        message: bytes, the data to be signed
        
    Returns:
        bytes, a generated signature based on the input data
        
    Raises:
        ValueError: if the specified hash algorithm is not supported
    """
    # Define secure algorithms (excluding weak ones like md5, sha1)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is supported and secure
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Get the hash function from hashlib
    try:
        hash_func = getattr(hashlib, algorithm_name)
    except AttributeError:
        raise ValueError(f"Hash algorithm '{algorithm_name}' not available in hashlib")
    
    # Create and return the HMAC signature
    return hmac.new(secret_key, message, hash_func).digest()
