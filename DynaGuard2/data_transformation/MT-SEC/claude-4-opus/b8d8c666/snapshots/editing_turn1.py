import hashlib

def generate_hash(input_string):
    """Generate SHA-256 hash of input string and return hexadecimal format."""
    return hashlib.sha256(input_string.encode()).hexdigest()
