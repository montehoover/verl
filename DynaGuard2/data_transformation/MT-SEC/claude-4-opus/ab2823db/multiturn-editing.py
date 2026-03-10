import hashlib

def hash_password(algo_name, secret):
    try:
        hash_object = hashlib.new(algo_name, secret.encode())
        return hash_object.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")
