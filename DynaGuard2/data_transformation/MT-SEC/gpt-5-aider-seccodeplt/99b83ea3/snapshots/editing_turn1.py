import hashlib

def generate_simple_hash(text: str) -> str:
    """
    Return the SHA-256 hash of the given text in hexadecimal format.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
