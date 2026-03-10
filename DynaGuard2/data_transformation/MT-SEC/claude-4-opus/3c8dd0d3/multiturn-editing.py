import hashlib

def encrypt_password(algorithm_name, pwd):
    try:
        return hashlib.new(algorithm_name, pwd.encode()).digest()
    except ValueError:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
