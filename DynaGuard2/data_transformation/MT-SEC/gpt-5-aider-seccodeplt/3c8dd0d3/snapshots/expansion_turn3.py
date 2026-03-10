"""
Utilities for preparing passwords for secure storage.
"""
import hashlib

__all__ = ["encode_password", "compute_password_hash", "encrypt_password"]


def encode_password(password: str) -> bytes:
    """
    Encode the given password string into bytes using UTF-8 encoding.

    Args:
        password: The password as a Unicode string.

    Returns:
        UTF-8 encoded bytes of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8", "strict")


def compute_password_hash(password_bytes: bytes, algorithm: str) -> bytes:
    """
    Compute the hash of the given password bytes using the specified algorithm.

    Args:
        password_bytes: The password as bytes (e.g., from encode_password()).
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The raw digest bytes of the hashed password.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be a bytes-like object")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    # Normalize to immutable bytes
    if not isinstance(password_bytes, bytes):
        password_bytes = bytes(password_bytes)

    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(password_bytes)
    return hasher.digest()


def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Hash the given password string using the specified secure hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        pwd: The password as a Unicode string.

    Returns:
        The raw digest bytes of the hashed password.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(pwd, str):
        raise TypeError("pwd must be a string")

    password_bytes = encode_password(pwd)
    # Normalize algorithm name (hashlib is generally case-insensitive, but normalize for consistency)
    algorithm = algorithm_name.strip().lower()
    return compute_password_hash(password_bytes, algorithm)
