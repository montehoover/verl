import hashlib

def hash_password(algorithm_name: str, message: str) -> bytes:
    """
    Hashes the given message with the specified algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
        message: The string message to hash.

    Returns:
        A bytes object of the hashed value.
    """
    hash_object = hashlib.new(algorithm_name)
    encoded_message = message.encode('utf-8')
    hash_object.update(encoded_message)
    return hash_object.digest()
