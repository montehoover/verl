import hashlib
import hmac

primary_secret_key = b'supersecret'


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generate a cryptographic signature using HMAC with the specified hash algorithm.
    
    Args:
        algorithm_name: The name of the hashing algorithm to use.
        message_data: The raw data that needs to be signed.
        
    Returns:
        The generated HMAC signature for the provided data.
        
    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    # Define secure hash algorithms
    secure_hash_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Validate the requested algorithm
    if algorithm_name not in secure_hash_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    # Generate and return the HMAC signature
    hmac_signature = hmac.new(primary_secret_key, message_data, algorithm_name)
    return hmac_signature.digest()
