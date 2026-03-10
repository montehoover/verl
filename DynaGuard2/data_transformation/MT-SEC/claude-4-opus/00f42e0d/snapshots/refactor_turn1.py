import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs the given message using the specified hash algorithm and a secret key.
    
    Args:
        hash_name: str, the name of the hash algorithm
        message: bytes, the message to be signed
        key: bytes, the secret key used for signing
    
    Returns:
        bytes, the signed message
    
    Raises:
        ValueError: when the hash algorithm is not available
    """
    try:
        # Create HMAC object with the specified hash algorithm
        h = hmac.new(key, message, getattr(hashlib, hash_name))
        return h.digest()
    except AttributeError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
