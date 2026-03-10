import hashlib

def simple_hash(input_string: str) -> str:
    """
    Return the SHA-256 hash of the given input string as a hexadecimal string.

    Parameters:
        input_string (str): The input string to hash.

    Returns:
        str: The SHA-256 hash of the input string in hexadecimal form.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()
