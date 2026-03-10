import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs the given message using the specified hash algorithm and a secret key.

    Args:
        hash_name: The name of the hash algorithm.
        message: The message to be signed.
        key: The secret key used for signing.

    Returns:
        The signed message (digest).

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available.")
    
    signer = hmac.new(key, message, getattr(hashlib, hash_name))
    return signer.digest()
