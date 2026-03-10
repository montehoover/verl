import hashlib

def generate_hash(input_text: str) -> str:
    """
    Generate a SHA-256 hash for the given input string.

    Args:
        input_text: The input string to hash.

    Returns:
        Hexadecimal string of the SHA-256 hash.
    """
    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string")
    return hashlib.sha256(input_text.encode('utf-8')).hexdigest()
