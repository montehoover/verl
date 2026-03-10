import hashlib


def hash_password(algorithm_name, message):
    """
    Hash the given message using the specified algorithm.

    Args:
        algorithm_name (str): Name of the hashing algorithm (e.g., 'sha256').
        message (str | bytes | bytearray | memoryview): Data to hash.

    Returns:
        bytes: The hash digest.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a str")
    algo = algorithm_name.lower()

    if isinstance(message, bytes):
        data = message
    elif isinstance(message, bytearray):
        data = bytes(message)
    elif isinstance(message, memoryview):
        data = message.tobytes()
    elif isinstance(message, str):
        data = message.encode("utf-8")
    else:
        raise TypeError("message must be of type str, bytes, bytearray, or memoryview")

    try:
        return hashlib.new(algo, data).digest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e
