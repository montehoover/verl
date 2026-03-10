import hashlib
import hmac

primary_secret_key = b'supersecret'

def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generates a cryptographic signature using HMAC with the specified hash algorithm.
    
    Args:
        algorithm_name: str, the name of the hashing algorithm to use
        message_data: bytes, the raw data that needs to be signed
        
    Returns:
        bytes, the generated signature for the provided data
        
    Raises:
        ValueError: if an unsupported or insecure hash algorithm is specified
    """
    # Define secure algorithms (excluding known weak ones like MD5, SHA1, etc.)
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 
        'sha512_256', 'sha3_224', 'sha3_256', 
        'sha3_384', 'sha3_512', 'blake2b', 'blake2s'
    }
    
    # Check if the algorithm is supported and secure
    if algorithm_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Get the hash algorithm from hashlib
    try:
        hash_algorithm = getattr(hashlib, algorithm_name)
    except AttributeError:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Create HMAC signature
    return hmac.new(primary_secret_key, message_data, hash_algorithm).digest()
