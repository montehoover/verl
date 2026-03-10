import hashlib
import hmac


def compute_basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string of the computed hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm name must be a string.")

    algo = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo)
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()


def generate_hmac_signature(message: bytes, algorithm: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC signature for the given message using the specified algorithm and secret key.

    Args:
        message: The message to authenticate as bytes.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').
        secret_key: The secret key as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm name must be a string.")

    algo = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    mac = hmac.new(secret_key, message, digestmod=algo)
    return mac.digest()
