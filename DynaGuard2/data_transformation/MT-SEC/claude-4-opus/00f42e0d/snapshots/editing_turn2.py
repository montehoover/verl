import hashlib

def sign_message(hash_name, message):
    try:
        hash_obj = hashlib.new(hash_name)
        hash_obj.update(message)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")
