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


def hash_with_salt(password: str, salt: str) -> str:
    """
    Hash a password combined with a salt using a strong algorithm (SHA-512).

    Args:
        password: The password to hash.
        salt: The salt to combine with the password.

    Returns:
        Hexadecimal string of the salted password hash.

    Raises:
        ValueError: If the chosen algorithm is not supported by hashlib.
    """
    algorithm = "sha512"
    try:
        hasher = hashlib.new(algorithm)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    # Concatenate salt and password in a deterministic order.
    # Using a delimiter reduces ambiguity in concatenation.
    combined = f"{salt}:{password}".encode("utf-8")
    hasher.update(combined)
    return hasher.hexdigest()
