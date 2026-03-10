import hashlib
import hmac
from typing import ByteString


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute the HMAC signature of payload using the given secret and hash algorithm.

    Args:
        hash_algorithm: The name of the hash function to apply (e.g., 'sha256').
        payload: The input bytes to be signed.
        secret: The secret key used for signing.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the hash function is not supported by hashlib or incompatible with HMAC.
    """
    if not isinstance(hash_algorithm, str):
        raise ValueError("hash_algorithm must be a string")

    # Normalize algorithm name for lookup
    algo = hash_algorithm.lower()

    # Check availability in hashlib
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash function: {hash_algorithm}")

    try:
        # hmac.new accepts a string digest name compatible with hashlib
        return hmac.new(secret, payload, algo).digest()
    except Exception as exc:
        # Re-raise as ValueError to match the specified contract
        raise ValueError(f"Unsupported or incompatible hash function for HMAC: {hash_algorithm}") from exc
