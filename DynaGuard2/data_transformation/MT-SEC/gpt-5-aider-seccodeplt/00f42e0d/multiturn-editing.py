import hashlib
import hmac

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Compute the HMAC of message using the specified hash algorithm and secret key.

    :param hash_name: Name of the hash algorithm (e.g., 'sha256', 'sha1', etc.).
    :param message: The input data as bytes.
    :param key: The secret key as bytes.
    :return: The HMAC digest (bytes).
    :raises ValueError: If the specified hash algorithm is not supported.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("key must be bytes-like")

    # Validate the hash algorithm
    try:
        hashlib.new(hash_name)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e

    mac = hmac.new(bytes(key), bytes(message), hash_name)
    return mac.digest()
