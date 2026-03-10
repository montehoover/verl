import hashlib
import hmac

key = b"supersecret"


def compute_message_hash(message: str, algorithm: str) -> bytes:
    """
    Compute the cryptographic hash of a message using the specified algorithm.

    Args:
        message: The input message as a string.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the specified algorithm is not supported.
        TypeError: If the inputs are not of the expected types.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a str")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a str")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(message.encode("utf-8"))
    return hasher.digest()


def create_hmac(message: bytes, algorithm: str) -> bytes:
    """
    Create an HMAC of the given message using the specified hash algorithm.

    Args:
        message: The input message as bytes.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The HMAC digest as bytes.

    Raises:
        ValueError: If the specified algorithm is not supported.
        TypeError: If the inputs are not of the expected types.
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a str")

    # Validate algorithm support explicitly
    try:
        hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    return hmac.new(key, message, algorithm).digest()


def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Sign the input message using HMAC with the specified hash algorithm and secret key.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        message: The message to sign as bytes.
        key: The secret key as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
        TypeError: If inputs are not of the expected types.
    """
    if not isinstance(hash_name, str):
        raise TypeError("hash_name must be a str")
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("key must be bytes-like")

    # Validate algorithm support explicitly
    try:
        hashlib.new(hash_name)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e

    return hmac.new(key, message, hash_name).digest()
