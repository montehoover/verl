import hashlib

def hash_password(hash_name, password):
    try:
        hash_object = hashlib.new(hash_name)
        hash_object.update(password.encode())
        return hash_object.digest()
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")
