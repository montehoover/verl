import hashlib

def simple_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()
