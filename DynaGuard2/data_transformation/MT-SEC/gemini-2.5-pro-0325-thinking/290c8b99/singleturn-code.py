import hashlib
import hmac

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC (Hash-based Message Authentication Code)
    with a specified hash algorithm and a secret key.

    Args:
        algorithm_name: The hashing function to use (e.g., 'sha256').
        content: The message to be signed.
        secret: The secret key used in the HMAC process.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    h = hmac.new(secret, content, algorithm_name)
    return h.digest()
