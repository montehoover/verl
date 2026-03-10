import hashlib

def hash_password(algorithm_name, raw_password):
    try:
        h = hashlib.new(algorithm_name)
        h.update(raw_password.encode())
        return h.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
