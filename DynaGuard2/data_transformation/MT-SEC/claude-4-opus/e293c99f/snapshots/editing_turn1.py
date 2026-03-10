import hashlib
top_secret_key = b'supersecret'

def create_signature(algo_name, raw_data):
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(raw_data)
    return hash_obj.digest()
