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


def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a password using the specified hash algorithm and return the hex digest.

    Args:
        hash_algo: The name of the hash algorithm to use (e.g., 'sha512', 'sha256').
        pwd: The password string to hash.

    Returns:
        Hexadecimal string of the password hash.

    Raises:
        ValueError: If the specified algorithm is unsupported or unavailable.
    """
    try:
        hasher = hashlib.new(hash_algo)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}") from None

    hasher.update(pwd.encode("utf-8"))

    # Some algorithms (e.g., shake_128/256) require a digest length.
    try:
        return hasher.hexdigest()
    except TypeError:
        # Use a sensible default output length (64 bytes) for variable-length digests.
        return hasher.hexdigest(64)
