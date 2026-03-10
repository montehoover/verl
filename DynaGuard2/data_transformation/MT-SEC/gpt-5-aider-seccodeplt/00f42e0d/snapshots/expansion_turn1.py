import hashlib


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
