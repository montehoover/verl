import hashlib
import hmac


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute an HMAC signature for the given payload using the provided secret and hash algorithm.

    Args:
        hash_algorithm: Name of the hash function to use (e.g., 'sha256').
        payload: Data to sign.
        secret: Secret key.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the hash function is not supported by hashlib.
    """
    if not isinstance(hash_algorithm, str):
        raise ValueError("hash_algorithm must be a string")

    available = {name.lower() for name in hashlib.algorithms_available}
    algo = hash_algorithm.lower()
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # hmac.new accepts a string digest name; hashlib will resolve it.
    return hmac.new(secret, payload, digestmod=algo).digest()
