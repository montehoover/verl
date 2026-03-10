import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data: str, algorithm: str) -> str:
    """
    Compute the hash of the input string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm: str) -> bytes:
    """
    Generate an HMAC for the given message using a predefined secret key and the specified algorithm.

    Args:
        message: The message to authenticate as bytes.
        algorithm: The hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The HMAC as raw bytes.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    try:
        # Validate the algorithm by attempting to construct a hasher
        hashlib.new(algorithm.lower())
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    h = hmac.new(secret_key, message, digestmod=lambda: hashlib.new(algorithm.lower()))
    return h.digest()
