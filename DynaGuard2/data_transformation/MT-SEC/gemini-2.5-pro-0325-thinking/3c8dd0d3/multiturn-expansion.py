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

def compute_password_hash(encoded_password: bytes, algorithm_name: str) -> str:
    """
    Computes the hash of an encoded password using the specified algorithm.

    Args:
        encoded_password: The byte-encoded password.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal representation of the password hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(encoded_password)
        return hasher.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Encrypts a password using the specified secure hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256').
        pwd: The password string to encrypt.

    Returns:
        The byte representation of the password hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    encoded_pwd = encode_password(pwd)
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(encoded_pwd)
        return hasher.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
