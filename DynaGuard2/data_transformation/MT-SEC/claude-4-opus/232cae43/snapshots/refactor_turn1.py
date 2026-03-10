import hashlib
import hmac

def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute the HMAC signature of the data.
    
    Args:
        hash_algorithm: str, the hash function to apply
        payload: bytes, the input to be signed
        secret: bytes, the secret key used for signing with the hash function
    
    Returns:
        bytes, the computed HMAC signature
    
    Raises:
        ValueError: if the hash function is not supported by hashlib
    """
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_algorithm}' is not supported by hashlib")
    
    return hmac.new(secret, payload, hash_algorithm).digest()
