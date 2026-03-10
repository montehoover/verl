import hashlib
from typing import Final


def compute_hash(data: str, algorithm: str) -> str:
    """
    Compute a hexadecimal hash digest of the given data using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The hash algorithm name (e.g., 'sha256', 'sha1', 'md5', 'sha3_256').

    Returns:
        Hexadecimal string of the hash digest.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
    """
    try:
        hasher = hashlib.new(algorithm)
    except (ValueError, TypeError):
        # Normalize the algorithm name in the message for consistency
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()
