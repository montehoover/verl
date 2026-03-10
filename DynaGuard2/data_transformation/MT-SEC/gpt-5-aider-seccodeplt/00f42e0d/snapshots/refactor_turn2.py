import hashlib
import hmac


def _validate_key(key: bytes) -> None:
    """
    Pure function to validate the signing key.

    :param key: The secret key for signing as a bytes-like object.
    :raises TypeError: If the key is not a bytes-like object.
    """
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("key must be a bytes-like object")


def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Sign the given message using HMAC with the specified hash algorithm and secret key.

    :param hash_name: The name of the hash algorithm (e.g., 'sha256').
    :param message: The message to be signed as bytes.
    :param key: The secret key for signing as bytes.
    :return: The signature as bytes.
    :raises ValueError: If the hash algorithm is not available.
    """
    # Guard: validate key early
    _validate_key(key)

    # Guard: validate hash algorithm availability early
    try:
        hashlib.new(hash_name)
    except Exception:
        raise ValueError(f"Hash algorithm not available: {hash_name}") from None

    # Compute HMAC digest
    return hmac.new(key, message, digestmod=hash_name).digest()
