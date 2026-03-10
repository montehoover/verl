import hashlib
import hmac

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a provided message using HMAC (Hash-based Message Authentication Code)
    with a specified hash algorithm and a secret key.

    Args:
        method_name: The hashing function to use (e.g., "sha256").
        data: The message to be signed.
        secret_key: The secret key used in the HMAC process.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available.
    """
    if method_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {method_name}")

    h = hmac.new(secret_key, data, method_name)
    return h.digest()
