import hashlib


def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    Parameters:
        password (str): The password to encode.

    Returns:
        bytes: The UTF-8 encoded bytes of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError(f"password must be a str, got {type(password).__name__}")
    return password.encode("utf-8")


def compute_password_hash(algorithm_name: str, encoded_password: bytes) -> bytes:
    """
    Compute the hash of an encoded password using the specified algorithm.

    Parameters:
        algorithm_name (str): The name of the hash algorithm (e.g., 'sha256').
        encoded_password (bytes): The password already encoded as bytes.

    Returns:
        bytes: The raw digest (bytes) of the hashed password.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError(f"algorithm_name must be a str, got {type(algorithm_name).__name__}")
    if not isinstance(encoded_password, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"encoded_password must be bytes-like (bytes, bytearray, memoryview), got {type(encoded_password).__name__}"
        )

    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from exc

    hasher.update(bytes(encoded_password))
    return hasher.digest()


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Securely compute the cryptographic hash of a password using the specified algorithm.

    Parameters:
        algorithm_name (str): The name of the hash algorithm (e.g., 'sha256').
        user_password (bytes): The user's password as bytes.

    Returns:
        bytes: The raw digest (bytes) of the hashed password.

    Raises:
        TypeError: If inputs are of incorrect type.
        ValueError: If the algorithm is unsupported.
    """
    return compute_password_hash(algorithm_name, user_password)
