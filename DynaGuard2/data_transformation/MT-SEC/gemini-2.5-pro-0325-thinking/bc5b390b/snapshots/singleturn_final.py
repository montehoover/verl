import hashlib

DISALLOWED_ALGORITHMS = {'md5', 'sha1', 'md5-sha1'}

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm (e.g., "sha256", "sha512").
        content: The input data to be hashed, as bytes.

    Returns:
        The resulting hash of the provided content, as bytes.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    if algorithm_name.lower() in DISALLOWED_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is disallowed due to security concerns."
        )

    try:
        # Check if the algorithm is available in hashlib.algorithms_available
        # hashlib.new() will also raise ValueError for an unknown algorithm,
        # but checking explicitly provides a clearer error path if needed.
        if algorithm_name.lower() not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm '{algorithm_name}' is not available.")
            
        hasher = hashlib.new(algorithm_name)
    except ValueError as e:
        # This catches cases where hashlib.new() itself fails for an algorithm
        # that might have passed the DISALLOWED_ALGORITHMS check but is still problematic
        # or truly unavailable in a way not caught by hashlib.algorithms_available
        # (though typically hashlib.algorithms_available is comprehensive).
        raise ValueError(
            f"Failed to initialize hash algorithm '{algorithm_name}': {e}"
        ) from e

    hasher.update(content)
    return hasher.digest()
