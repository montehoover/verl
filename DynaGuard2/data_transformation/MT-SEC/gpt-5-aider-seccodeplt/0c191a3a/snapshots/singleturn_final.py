import hashlib
import hmac
from functools import partial

# Secret key used for signing the data
secret_key = b'supersecret'


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generate a digital signature (HMAC) for the given data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256').
        data: The data to be signed.

    Returns:
        The digital signature of the data as bytes.

    Raises:
        ValueError: If the hash algorithm is not available or not suitable for HMAC.
    """
    # Validate that the algorithm exists
    try:
        hasher = hashlib.new(hash_name)
    except (ValueError, TypeError):
        raise ValueError(f"Hash algorithm '{hash_name}' is not available.")

    # Ensure the algorithm has a fixed digest size (required by HMAC; SHAKE variants won't work)
    digest_size = getattr(hasher, "digest_size", 0)
    if not isinstance(digest_size, int) or digest_size <= 0:
        raise ValueError(f"Hash algorithm '{hash_name}' is not supported for HMAC.")

    # Use hashlib.new via partial to support algorithms not exposed as direct attributes
    digestmod = partial(hashlib.new, hash_name)
    return hmac.new(secret_key, data, digestmod=digestmod).digest()
