import hashlib

def sign_message(hash_name, message):
    hash_obj = hashlib.new(hash_name)
    hash_obj.update(message)
    return hash_obj.digest()
