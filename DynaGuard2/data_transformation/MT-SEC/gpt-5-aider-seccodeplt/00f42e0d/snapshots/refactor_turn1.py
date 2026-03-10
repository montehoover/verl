import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Sign the given message using HMAC with the specified hash algorithm and secret key.

    :param hash_name: The name of the hash algorithm (e.g., 'sha256').
    :param message: The message to be signed as bytes.
    :param key: The secret key for signing as bytes.
    :return: The signature as bytes.
    :raises ValueError: If the hash algorithm is not available.
    """
    try:
        mac = hmac.new(key, message, digestmod=hash_name)
    except Exception:
        raise ValueError(f"Hash algorithm not available: {hash_name}") from None
    return mac.digest()
