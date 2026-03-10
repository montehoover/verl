import hashlib

def hash_password(algorithm_name, pwd):
    return hashlib.new(algorithm_name, pwd.encode()).digest()
