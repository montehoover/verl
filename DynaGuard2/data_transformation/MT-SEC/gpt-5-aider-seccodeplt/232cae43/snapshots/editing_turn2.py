import hashlib
import hmac


def generate_hash(text: str) -> str:
    """
    Generate the SHA-256 hash of the given text.

    Args:
        text: The input string to hash.

    Returns:
        The SHA-256 hash as a hexadecimal string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_hmac(input_data: bytes, secret_key: bytes) -> bytes:
    """
    Generate an HMAC signature using SHA-256 for the given input data and secret key.

    Args:
        input_data: The message data to authenticate, as bytes.
        secret_key: The secret key used for HMAC, as bytes.

    Returns:
        The HMAC-SHA256 signature as raw bytes.
    """
    if not isinstance(input_data, (bytes, bytearray)):
        raise TypeError("input_data must be bytes or bytearray")
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key must be bytes or bytearray")

    return hmac.new(secret_key, input_data, hashlib.sha256).digest()
