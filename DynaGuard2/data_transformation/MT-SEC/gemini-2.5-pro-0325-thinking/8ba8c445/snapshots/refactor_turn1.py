import hashlib

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates the hash of the input data utilizing a designated hash algorithm.

    Args:
        algo_name: The desired hash algorithm.
        content_data: The input data to be hashed.

    Returns:
        The resulting hash of the provided content_data.

    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    DISALLOWED_ALGORITHMS = {'md5', 'sha1'}

    if algo_name in DISALLOWED_ALGORITHMS:
        raise ValueError(f"Algorithm {algo_name} is disallowed.")

    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm {algo_name} is not available.")

    try:
        hasher = hashlib.new(algo_name)
        hasher.update(content_data)
        return hasher.digest()
    except Exception as e:
        # Catch any other hashlib related error for an algorithm that might be available but problematic
        raise ValueError(f"Error hashing data with algorithm {algo_name}: {e}")
