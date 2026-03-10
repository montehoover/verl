import hashlib

def encode_password(password: str) -> bytes:
    """
    Encodes a password string into its byte representation.

    Args:
        password: The password string to encode.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')

def compute_hash(encoded_password: bytes, algorithm_name: str) -> str:
    """
    Computes the hash value of an encoded password using the specified algorithm.

    Args:
        encoded_password: The byte-encoded password.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the password.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(encoded_password)
        return hasher.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Generates a cryptographic hash for a given password using secure hash algorithms.

    Args:
        algo_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        pw: The password string.

    Returns:
        The cryptographic hash as bytes.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    encoded_pw = encode_password(pw)
    try:
        hasher = hashlib.new(algo_name)
        hasher.update(encoded_pw)
        return hasher.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")
