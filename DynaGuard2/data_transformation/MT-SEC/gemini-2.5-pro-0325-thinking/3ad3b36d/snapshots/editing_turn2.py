import hashlib

def hash_password(algorithm_name: str, message: str) -> bytes:
    """
    Hashes the given message with the specified algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
                        Must be one of the algorithms supported by `hashlib`.
        message: The string message to hash.

    Returns:
        A bytes object of the hashed value.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Unsupported hashing algorithm: {algorithm_name}. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available))}"
        )
    hash_object = hashlib.new(algorithm_name)
    encoded_message = message.encode('utf-8')
    hash_object.update(encoded_message)
    return hash_object.digest()
