import hmac
import hashlib

def generate_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC-SHA256 signature for the given input string using the provided secret key.

    Args:
        input_string (str): The message data to sign.
        secret_key (bytes): The secret key used for HMAC.

    Returns:
        bytes: The HMAC-SHA256 signature as raw bytes.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")
    if not isinstance(secret_key, bytes):
        raise TypeError("secret_key must be bytes")
    return hmac.new(secret_key, input_string.encode('utf-8'), hashlib.sha256).digest()
