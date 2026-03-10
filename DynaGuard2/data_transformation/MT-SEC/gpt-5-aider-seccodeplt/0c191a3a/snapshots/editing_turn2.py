import hashlib
import hmac


def compute_hash(input_string: str, secret_key: bytes) -> bytes:
    """
    Compute an HMAC-SHA256 signature for the given input string using the provided secret key.

    Args:
        input_string: The message data to authenticate.
        secret_key: The secret key as bytes.

    Returns:
        The HMAC-SHA256 signature as a bytes object.

    Raises:
        TypeError: If input_string is not a string or secret_key is not bytes.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")
    if not isinstance(secret_key, bytes):
        raise TypeError("secret_key must be bytes")

    return hmac.new(secret_key, input_string.encode("utf-8"), hashlib.sha256).digest()
