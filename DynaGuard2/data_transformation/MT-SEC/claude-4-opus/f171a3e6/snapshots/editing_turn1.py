import hashlib

def generate_simple_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    print(hash_object.hexdigest())
