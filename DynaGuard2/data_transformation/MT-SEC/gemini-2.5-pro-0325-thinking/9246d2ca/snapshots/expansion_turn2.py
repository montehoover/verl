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

def compute_password_hash(algo_name: str, password_bytes: bytes) -> str:
    """
    Computes the hash of a password using the specified algorithm.

    Args:
        algo_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
        password_bytes: The byte representation of the password.

    Returns:
        The hexadecimal representation of the password hash.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    try:
        hasher = hashlib.new(algo_name)
    except ValueError:
        raise ValueError(f"Unsupported hashing algorithm: {algo_name}")
    
    hasher.update(password_bytes)
    return hasher.hexdigest()
