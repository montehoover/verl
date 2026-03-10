import hashlib

def generate_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()
