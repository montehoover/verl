import hashlib

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()
