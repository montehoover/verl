import hashlib


def generate_hash(text: str) -> str:
    """
    Generate the SHA-256 hash of the given text.

    Args:
        text: The input string to hash.

    Returns:
        The SHA-256 hash as a hexadecimal string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
