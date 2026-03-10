import hashlib

def generate_simple_hash(input_string):
    """Generate SHA-256 hash of input string and return in hexadecimal format."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()
