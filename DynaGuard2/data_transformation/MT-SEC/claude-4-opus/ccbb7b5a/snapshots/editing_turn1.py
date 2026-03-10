import hashlib

def generate_hash(input_string):
    """Generate SHA-256 hash of input string and return as hexadecimal string."""
    hash_object = hashlib.sha256(input_string.encode())
    return hash_object.hexdigest()
