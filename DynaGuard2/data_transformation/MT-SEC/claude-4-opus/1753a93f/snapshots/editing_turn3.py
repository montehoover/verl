import hashlib

def encrypt_pwd(hash_algo, pwd):
    try:
        hash_func = hashlib.new(hash_algo)
        hash_func.update(pwd.encode())
        return hash_func.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_algo}")
