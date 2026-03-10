import hashlib

def generate_hash(text: str) -> str:
    """
    Return the SHA-256 hexadecimal hash of the given text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
