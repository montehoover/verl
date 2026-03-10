import hashlib
import hmac

def generate_simple_hash(input_str: str) -> str:
    """
    Generate a SHA-256 hash of the given string and return its hexadecimal representation.

    Args:
        input_str: The input string to hash.

    Returns:
        A hex string of the SHA-256 digest.
    """
    if not isinstance(input_str, str):
        raise TypeError(f"input_str must be of type str, got {type(input_str).__name__}")
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()


def generate_signed_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature of the given string using the provided secret key.

    Args:
        input_string: The message data to sign.
        secret_key: The secret key as bytes.

    Returns:
        The HMAC-SHA256 signature as bytes.
    """
    if not isinstance(input_string, str):
        raise TypeError(f"input_string must be of type str, got {type(input_string).__name__}")
    if not isinstance(secret_key, (bytes, bytearray, memoryview)):
        raise TypeError(f"secret_key must be bytes-like, got {type(secret_key).__name__}")

    message_bytes = input_string.encode("utf-8")
    key_bytes = bytes(secret_key)
    return hmac.new(key_bytes, message_bytes, hashlib.sha256).digest()
