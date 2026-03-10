import hashlib

def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    This function does not perform normalization or trimming, preserving
    the exact character sequence provided by the user.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")


def compute_hash(password_bytes: bytes, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of a byte-encoded password using the specified algorithm.

    :param password_bytes: The password as bytes.
    :param algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').
    :return: Hexadecimal digest string.
    :raises ValueError: If the algorithm is unsupported.
    """
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be a bytes-like object")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    try:
        hasher = hashlib.new(algorithm.lower())
    except Exception:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(password_bytes)
    return hasher.hexdigest()
