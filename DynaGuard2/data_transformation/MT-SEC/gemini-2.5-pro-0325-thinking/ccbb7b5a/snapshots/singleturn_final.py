import hashlib

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of a given byte sequence using the hash algorithm specified.
    Avoids the usage of weak algorithms like md5 and sha1.

    Args:
        algorithm_name: The name of the hash algorithm to use.
        content: Byte-like object representing the input data.

    Returns:
        Bytes representing the generated hash value.

    Raises:
        ValueError: If the chosen hash algorithm is either unavailable or unsafe to use.
    """
    weak_algorithms = {'md5', 'sha1', 'md5-sha1'}

    if algorithm_name.lower() in weak_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is unsafe and therefore not allowed.")

    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available in hashlib.")

    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(content)
        return hasher.digest()
    except Exception as e:
        # Catch any other hashlib related error during instantiation or update
        raise ValueError(f"Error using algorithm '{algorithm_name}': {e}")
