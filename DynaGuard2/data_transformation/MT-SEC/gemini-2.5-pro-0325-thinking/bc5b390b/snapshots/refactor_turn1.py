import hashlib

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha512').
        content: The input data to be hashed.

    Returns:
        The resulting hash of the provided content.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    disallowed_algorithms = {'md5', 'sha1'}

    if algorithm_name in disallowed_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is disallowed due to security concerns.")

    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available in hashlib.")

    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(content)
        return hasher.digest()
    except Exception as e:
        # Catch any other hashlib related error during new() or update()
        # and wrap it in a ValueError for consistency, though hashlib.new
        # should raise ValueError for unknown algorithms if not caught by algorithms_available.
        raise ValueError(f"Error generating hash with algorithm '{algorithm_name}': {e}")
