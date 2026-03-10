import hashlib

def encode_password(password: str) -> bytes:
    """
    Encodes a password string into bytes.

    Args:
        password: The password string to encode.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')

def compute_hash(encoded_password: bytes, algorithm_name: str) -> str:
    """
    Computes the hash of an encoded password using the specified algorithm.

    Args:
        encoded_password: The byte-encoded password.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hash value in hexadecimal format.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(encoded_password)
        return hasher.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
