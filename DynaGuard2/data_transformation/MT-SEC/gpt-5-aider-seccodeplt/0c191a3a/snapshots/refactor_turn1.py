import hashlib
import hmac

# Secret key used for signing the data
secret_key = b'supersecret'


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generate a digital signature (HMAC) for the given data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256").
        data: The data to be signed as bytes.

    Returns:
        The digital signature of the data as bytes.

    Raises:
        ValueError: If the hash algorithm is not available or unsupported for HMAC.
    """
    try:
        h = hmac.new(secret_key, data, digestmod=hash_name)
        return h.digest()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Hash algorithm not available or unsupported for HMAC: {hash_name}") from e
