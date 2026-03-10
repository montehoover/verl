import hashlib


def _validate_algorithm(algorithm_name: str) -> None:
    """
    Validates the hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    disallowed_algorithms = {'md5', 'sha1'}
    if algorithm_name in disallowed_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is disallowed due to security concerns.")

    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available in hashlib.")


def _perform_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Performs the hashing operation.

    Args:
        algorithm_name: The desired hash algorithm.
        content: The input data to be hashed.

    Returns:
        The resulting hash of the provided content.

    Raises:
        ValueError: If an error occurs during hashing.
    """
    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(content)
        return hasher.digest()
    except Exception as e:
        # Catch any other hashlib related error during new() or update()
        # and wrap it in a ValueError for consistency.
        raise ValueError(f"Error generating hash with algorithm '{algorithm_name}': {e}")


def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha512').
        content: The input data to be hashed.

    Returns:
        The resulting hash of the provided content.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed,
                    or if an error occurs during hashing.
    """
    _validate_algorithm(algorithm_name)
    return _perform_hash(algorithm_name, content)
