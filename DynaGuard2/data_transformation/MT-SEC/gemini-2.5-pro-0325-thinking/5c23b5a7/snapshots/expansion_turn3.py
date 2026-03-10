import hashlib

def encode_password(password: str) -> bytes:
    """
    Encodes a password string into bytes.

    Args:
        password: The password string to encode.

    Returns:
        The password encoded as bytes.
    """
    return password.encode('utf-8')


def compute_password_hash(algorithm_name: str, encoded_password: bytes) -> str:
    """
    Computes the hash value of an encoded password using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        encoded_password: The password encoded as bytes.

    Returns:
        The hexadecimal representation of the password hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(encoded_password)
        return hasher.hexdigest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e


def secure_hash_password(algorithm_name: str, user_password: bytes) -> str:
    """
    Computes the cryptographic hash of a password using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        user_password: The password as bytes.

    Returns:
        The hexadecimal representation of the password hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(user_password)
        return hasher.hexdigest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e
