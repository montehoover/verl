import hashlib
import hmac

def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Computes the HMAC signature of the data.

    Args:
        hash_algorithm: The hash function to apply.
        payload: The input to be signed.
        secret: The secret key used for signing with the hash function.

    Returns:
        The computed HMAC signature.

    Raises:
        ValueError: If the hash function is not supported by hashlib.
    """
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    h = hmac.new(secret, payload, getattr(hashlib, hash_algorithm))
    return h.digest()
