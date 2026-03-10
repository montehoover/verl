import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs a message using HMAC with a specified hash algorithm and secret key.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        message: The input data as bytes.
        key: The secret key as bytes.

    Returns:
        The HMAC signed message as bytes.
    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    try:
        # Check if the hash algorithm is available via hashlib,
        # as hmac.new will use it.
        hashlib.new(hash_name)
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available for HMAC.")
    
    h = hmac.new(key, message, getattr(hashlib, hash_name))
    return h.digest()
