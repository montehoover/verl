import hashlib

def generate_hash(algorithm_name, content):
    if algorithm_name in hashlib.algorithms_available:
        hash_obj = hashlib.new(algorithm_name)
        hash_obj.update(content)
        return hash_obj.digest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
