import hashlib

def generate_simple_hash(input_str: str) -> str:
    """
    Generate a SHA-256 hash of the given string and return its hexadecimal representation.

    Args:
        input_str: The input string to hash.

    Returns:
        A hex string of the SHA-256 digest.
    """
    if not isinstance(input_str, str):
        raise TypeError(f"input_str must be of type str, got {type(input_str).__name__}")
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()
