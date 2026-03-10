import hashlib

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes the given password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256", "sha512").
        password: The password string to be hashed.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available. Supported algorithms: {hashlib.algorithms_available}")

    hasher = hashlib.new(hash_name)
    # Encode the password string to bytes before hashing
    # The problem description specifies password as str, while the example uses bytes.
    # Standard practice is to hash bytes, so we encode the string.
    hasher.update(password.encode('utf-8'))
    return hasher.digest()
