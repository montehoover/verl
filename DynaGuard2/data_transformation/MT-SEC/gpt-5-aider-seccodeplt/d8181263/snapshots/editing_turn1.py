import hashlib

def generate_hash(input_string: str) -> str:
    """
    Generate a SHA-256 hexadecimal hash for the given input string.

    Args:
        input_string (str): The input string to hash.

    Returns:
        str: The SHA-256 hash of the input string in hexadecimal format.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()
