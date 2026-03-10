import hashlib

def generate_hash(algorithm_name, data, connection):
    try:
        hash_object = hashlib.new(algorithm_name)
        hash_object.update(data.encode())
        hash_bytes = hash_object.digest()
        connection.send(hash_bytes)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
