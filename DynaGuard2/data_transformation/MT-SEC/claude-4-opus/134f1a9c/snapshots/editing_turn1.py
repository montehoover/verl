import hashlib

def derive_hash(algorithm_name, input_data):
    hash_obj = hashlib.new(algorithm_name)
    hash_obj.update(input_data)
    return hash_obj.digest()
