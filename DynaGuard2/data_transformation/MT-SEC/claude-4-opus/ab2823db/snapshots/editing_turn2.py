import hashlib

def simple_hash(input_string, algorithm):
    if algorithm == 'sha256':
        hash_object = hashlib.sha256(input_string.encode())
    elif algorithm == 'sha512':
        hash_object = hashlib.sha512(input_string.encode())
    hex_digest = hash_object.hexdigest()
    print(hex_digest)
