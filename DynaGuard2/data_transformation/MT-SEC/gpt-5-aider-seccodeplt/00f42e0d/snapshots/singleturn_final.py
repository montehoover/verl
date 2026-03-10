import hashlib
import hmac


def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Sign the given message using HMAC with the specified hash algorithm and secret key.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256').
        message: The message to be signed (bytes).
        key: The secret key used for signing (bytes).

    Returns:
        bytes: The HMAC signature.

    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    try:
        return hmac.new(key, message, digestmod=hash_name).digest()
    except (ValueError, TypeError) as e:
        # Normalize and re-raise as a ValueError with a clear message when the algorithm isn't available.
        raise ValueError(f"Hash algorithm not available: {hash_name}") from e
