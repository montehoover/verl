import hashlib

def encrypt_pwd(hash_algo, pwd):
    hash_func = hashlib.new(hash_algo)
    hash_func.update(pwd.encode())
    return hash_func.digest()
