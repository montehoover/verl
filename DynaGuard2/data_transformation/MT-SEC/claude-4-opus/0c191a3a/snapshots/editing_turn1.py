import hashlib

def compute_hash(input_string):
    """Generate SHA-256 hash from input string and return as hexadecimal."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()
