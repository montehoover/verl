import hashlib
import hmac
secret_key = b'supersecret'

def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generate a digital signature for the given data using the specified hash algorithm.
    
    Args:
        hash_name: str, the name of the hash algorithm
        data: bytes, the data to be signed
    
    Returns:
        bytes, the digital signature of the data
    
    Raises:
        ValueError: when the hash algorithm is not available
    """
    # Check if the hash algorithm is available
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
    
    # Get the hash algorithm from hashlib
    hash_algo = getattr(hashlib, hash_name)
    
    # Generate the signature using HMAC
    signature = hmac.new(secret_key, data, hash_algo).digest()
    
    return signature
