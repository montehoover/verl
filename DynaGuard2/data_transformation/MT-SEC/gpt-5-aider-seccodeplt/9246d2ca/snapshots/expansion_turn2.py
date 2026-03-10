import hashlib


def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8 encoding.

    Args:
        password: The password as a string.

    Returns:
        Bytes representation of the password using UTF-8.

    Raises:
        TypeError: If password is not a str.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    return password.encode("utf-8")


def compute_password_hash(algo_name: str, password_bytes: bytes) -> bytes:
    """
    Compute the cryptographic hash of the given password bytes using the specified algorithm.

    Args:
        algo_name: The name of the hash algorithm (e.g., 'sha256', 'sha512', 'md5').
        password_bytes: The password in bytes.

    Returns:
        The raw hash digest as bytes.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a str")
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be bytes-like")

    algo = algo_name.lower()
    try:
        hasher = hashlib.new(algo)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")

    hasher.update(bytes(password_bytes))
    return hasher.digest()
