import hashlib
from typing import Any

__all__ = ["encode_password", "compute_hash"]

def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    Parameters:
        password (str): The password as a Unicode string.

    Returns:
        bytes: The UTF-8 encoded byte representation of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    return password.encode("utf-8")


def compute_hash(hash_name: str, data: bytes) -> bytes:
    """
    Compute the hash of the given data using the specified hash algorithm.

    Parameters:
        hash_name (str): The name of the hash algorithm (e.g., 'sha256').
        data (bytes): The byte data to hash.

    Returns:
        bytes: The raw digest (hash) bytes.

    Raises:
        TypeError: If hash_name is not a string or data is not bytes-like.
        ValueError: If the hash algorithm is unsupported.
    """
    if not isinstance(hash_name, str):
        raise TypeError("hash_name must be a str")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    try:
        h = hashlib.new(hash_name)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e

    h.update(bytes(data))
    return h.digest()
