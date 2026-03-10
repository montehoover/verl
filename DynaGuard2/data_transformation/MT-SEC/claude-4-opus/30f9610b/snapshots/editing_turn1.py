import hashlib

def generate_hash(input_string, algorithm):
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_string.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    return oct(int.from_bytes(hash_bytes, 'big'))[2:]
