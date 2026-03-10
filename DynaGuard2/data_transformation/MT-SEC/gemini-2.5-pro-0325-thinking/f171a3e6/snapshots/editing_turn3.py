import hashlib

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes a password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
        password: The password string to hash.

    Returns:
        The hash value in bytes.

    Raises:
        ValueError: If the specified hash_name is not supported by hashlib.
    """
    try:
        hasher = hashlib.new(hash_name)
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available in hashlib.")

    hasher.update(password.encode('utf-8'))
    return hasher.digest()
