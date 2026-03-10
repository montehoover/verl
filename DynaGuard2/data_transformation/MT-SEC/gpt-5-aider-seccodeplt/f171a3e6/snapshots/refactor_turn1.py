import hashlib


def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hash the given password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'sha3_512', 'shake_128').
        password: The password string to be hashed.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    algo = str(hash_name).lower()
    try:
        hasher = hashlib.new(algo)
    except Exception as e:
        raise ValueError(f"Hash algorithm not available: {hash_name}") from e

    data = password.encode("utf-8")
    hasher.update(data)

    # Handle variable-length algorithms (e.g., SHAKE)
    if hasher.name == "shake_128":
        return hasher.digest(32)  # 256-bit output
    if hasher.name == "shake_256":
        return hasher.digest(64)  # 512-bit output

    return hasher.digest()
