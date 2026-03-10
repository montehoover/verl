import hashlib


def generate_simple_hash(text: str) -> str:
    """
    Return the SHA-256 hash of the input string in hexadecimal format.

    :param text: Input string to hash.
    :return: Hexadecimal string of the SHA-256 digest.
    """
    if not isinstance(text, str):
        raise TypeError("generate_simple_hash expects a string input")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
