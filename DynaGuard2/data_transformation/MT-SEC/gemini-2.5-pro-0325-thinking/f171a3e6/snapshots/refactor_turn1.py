import hashlib

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm.
        password: The password to be hashed.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    hasher = hashlib.new(hash_name)
    hasher.update(password.encode('utf-8'))
    return hasher.digest()
