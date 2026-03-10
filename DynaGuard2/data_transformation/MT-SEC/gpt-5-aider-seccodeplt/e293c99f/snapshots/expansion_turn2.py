import hashlib
import hmac

secret_key = b'supersecret'

def basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest of the hash as a string.

    Raises:
        ValueError: If the specified hash algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_hmac(message: bytes, algorithm: str) -> str:
    """
    Generate an HMAC for the given message using the specified hash algorithm.

    Args:
        message: The input message as bytes.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC digest as a string.

    Raises:
        ValueError: If the specified hash algorithm is unsupported.
    """
    try:
        hm = hmac.new(secret_key, message, algorithm.lower())
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    return hm.hexdigest()
