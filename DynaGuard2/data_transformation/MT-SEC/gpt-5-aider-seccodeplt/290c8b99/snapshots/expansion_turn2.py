import hashlib
import hmac


def compute_hash(message: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of a message using the specified algorithm.

    Args:
        message: The input text to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5', 'blake2b').

    Returns:
        The hexadecimal digest string of the computed hash.

    Raises:
        ValueError: If the algorithm is unsupported or requires a variable-length output (e.g., 'shake_*').
    """
    if not isinstance(message, str):
        raise TypeError("message must be a str")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a str")

    normalized_alg = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}

    # Exclude variable-length digest algorithms since they require an explicit length.
    if normalized_alg.startswith("shake_"):
        raise ValueError(f"Unsupported hash algorithm (variable-length digest): {algorithm}")

    if normalized_alg not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(normalized_alg)
    hasher.update(message.encode("utf-8"))
    return hasher.hexdigest()


def create_hmac(algorithm_name: str, message: bytes, secret_key: bytes) -> bytes:
    """
    Generate an HMAC for the given message using the provided secret key and hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha1', 'blake2b').
        message: The message as bytes.
        secret_key: The secret key as bytes.

    Returns:
        The HMAC as raw bytes.

    Raises:
        ValueError: If the hash algorithm is unsupported or has a variable-length digest (e.g., 'shake_*').
        TypeError: If inputs are not of the expected types.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a str")
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key must be bytes")

    normalized_alg = algorithm_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}

    # Exclude variable-length digest algorithms since they require an explicit length.
    if normalized_alg.startswith("shake_"):
        raise ValueError(f"Unsupported hash algorithm (variable-length digest): {algorithm_name}")

    if normalized_alg not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Ensure inputs are bytes (convert bytearray if provided)
    msg_bytes = bytes(message)
    key_bytes = bytes(secret_key)

    mac = hmac.new(key_bytes, msg_bytes, normalized_alg)
    return mac.digest()
