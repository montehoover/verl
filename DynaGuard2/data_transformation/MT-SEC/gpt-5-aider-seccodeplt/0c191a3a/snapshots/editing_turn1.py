import hashlib


def compute_hash(input_string: str) -> str:
    """
    Compute the SHA-256 hexadecimal hash of the given input string.

    Args:
        input_string: The input string to hash.

    Returns:
        A hexadecimal string representing the SHA-256 hash.

    Raises:
        TypeError: If input_string is not a string.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")

    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()
