import hashlib

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of a given byte sequence using the hash algorithm specified.
    Avoids the usage of weak algorithms like md5 and sha1.

    Args:
        algorithm_name: str, the name of the hash algorithm to use.
        content: bytes, byte-like object representing the input data.

    Returns:
        bytes, representing the generated hash value.

    Raises:
        ValueError: if the chosen hash algorithm is either unavailable or unsafe to use.
    """
    WEAK_ALGORITHMS = {"md5", "sha1", "md5-sha1"}

    if algorithm_name.lower() in WEAK_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is unsafe and therefore not allowed."
        )

    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not available in hashlib."
        )

    try:
        hasher = hashlib.new(algorithm_name)
        hasher.update(content)
        return hasher.digest()
    except Exception as e:
        # Catch any other hashlib related error during new() or update()
        # though the previous checks should cover most cases.
        raise ValueError(f"Error computing digest with '{algorithm_name}': {e}")
