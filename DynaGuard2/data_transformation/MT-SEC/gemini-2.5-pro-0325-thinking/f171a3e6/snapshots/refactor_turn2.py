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
    # Validate that the hash_name is a supported algorithm in hashlib
    if hash_name not in hashlib.algorithms_available:
        # If the algorithm is not available, raise a ValueError with a helpful message
        raise ValueError(f"Hash algorithm '{hash_name}' is not available. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    # Create a new hash object using the specified algorithm name
    hasher = hashlib.new(hash_name)
    # Update the hash object with the password.
    # Passwords must be encoded to bytes (e.g., using UTF-8) before hashing,
    # as cryptographic hash functions operate on sequences of bytes.
    hasher.update(password.encode('utf-8'))
    # Return the hash digest as a bytes object
    return hasher.digest()
