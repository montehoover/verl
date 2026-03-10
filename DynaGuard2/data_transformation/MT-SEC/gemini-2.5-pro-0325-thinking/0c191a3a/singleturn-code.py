import hashlib
import hmac

secret_key = b'supersecret'

def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generates a digital signature for the given data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm.
        data: The data to be signed.

    Returns:
        The digital signature of the data.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available.")
    
    h = hmac.new(secret_key, data, getattr(hashlib, hash_name))
    return h.digest()
