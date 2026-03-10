import hashlib
import hmac

secret_key = b'supersecret'


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


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generate an HMAC signature for the given data using the provided hash algorithm name
    and the global secret key.

    Args:
        hash_name: Name of the hash algorithm to use (e.g., 'sha256', 'sha512', etc.).
        data: Raw data to sign as bytes.

    Returns:
        The HMAC signature as a bytes object.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the hash algorithm is not available or unsupported for HMAC.
    """
    if not isinstance(hash_name, str):
        raise TypeError("hash_name must be a string")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    normalized_name = hash_name.lower()
    available = {alg.lower() for alg in hashlib.algorithms_available}
    if normalized_name not in available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")

    try:
        return hmac.new(secret_key, data, normalized_name).digest()
    except (TypeError, ValueError) as e:
        # Some algorithms (e.g., SHAKE) are not suitable for HMAC
        raise ValueError(f"Hash algorithm '{hash_name}' is not supported for HMAC") from e
