import hashlib

def encrypt_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()
