import hashlib

def hash_password(algorithm_name, message):
    h = hashlib.new(algorithm_name)
    h.update(message.encode())
    return h.digest()
