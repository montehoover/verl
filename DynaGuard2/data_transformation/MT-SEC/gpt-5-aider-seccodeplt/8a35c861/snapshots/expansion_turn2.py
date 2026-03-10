import hashlib

def encode_password(password: str) -> bytes:
    """
    Encode a password string to bytes using UTF-8.

    Args:
        password: The password as a string.

    Returns:
        The UTF-8 encoded byte representation of the password.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")


def compute_hash(password_bytes: bytes, algorithm: str) -> str:
    """
    Compute the hexadecimal hash digest for a byte-encoded password.

    Args:
        password_bytes: The password as bytes.
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal string digest of the hash.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the hash algorithm is unsupported.
    """
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be bytes-like")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(password_bytes)
    return hasher.hexdigest()
