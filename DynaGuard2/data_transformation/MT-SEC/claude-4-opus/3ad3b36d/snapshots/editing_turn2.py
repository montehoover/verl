import hashlib

def hash_password(algorithm_name, message):
    try:
        h = hashlib.new(algorithm_name)
        h.update(message.encode())
        return h.digest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
