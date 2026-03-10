import hashlib

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hashes the given raw password with the specified algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
                        Must be one of the algorithms supported by `hashlib`.
        raw_password: The string password to hash.

    Returns:
        A hexadecimal string representation of the hashed password.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Unsupported hashing algorithm: {algorithm_name}. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available))}"
        )
    hash_object = hashlib.new(algorithm_name)
    encoded_password = raw_password.encode('utf-8')
    hash_object.update(encoded_password)
    return hash_object.hexdigest()
