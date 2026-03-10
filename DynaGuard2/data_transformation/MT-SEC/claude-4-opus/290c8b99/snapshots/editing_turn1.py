import hashlib

def generate_signature(algorithm_name, content):
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(content)
    return hash_obj.digest()
