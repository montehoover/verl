import hashlib

def generate_signature(algorithm_name, content, secret):
    try:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(secret + content)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
