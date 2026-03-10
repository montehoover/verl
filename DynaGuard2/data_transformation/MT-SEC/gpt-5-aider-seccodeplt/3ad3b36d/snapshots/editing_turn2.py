import hashlib


def hash_password(algorithm_name, message):
    """
    Hash the given message using the specified algorithm.

    Args:
        algorithm_name (str): Name of the hashing algorithm (e.g., 'sha256').
        message (str | bytes | bytearray | memoryview): Data to hash.

    Returns:
        bytes: The hash digest.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a str")
    algo = algorithm_name.lower()

    # Prepare data as bytes
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

    # Determine supported algorithms, excluding variable-length digest algorithms (SHAKE)
    supported_algorithms = {a.lower() for a in hashlib.algorithms_available}
    unsupported_variable_length = {"shake_128", "shake_256"}
    supported_algorithms -= unsupported_variable_length

    if algo not in supported_algorithms:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm_name}. "
            f"Supported algorithms: {', '.join(sorted(supported_algorithms))}"
        )

    try:
        return hashlib.new(algo, data).digest()
    except ValueError as e:
        # Fallback in case environment-specific issues arise with hashlib.new
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e
