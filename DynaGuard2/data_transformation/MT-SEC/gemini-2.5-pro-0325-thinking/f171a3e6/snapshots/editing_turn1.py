import hashlib

def generate_simple_hash(input_string: str):
    """
    Generates and prints the SHA-256 hash of a given string.

    Args:
        input_string: The string to hash.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    print(sha256_hash)
