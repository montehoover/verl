import hashlib

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    # List of unsafe algorithms to disallow
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is unsafe
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Unsafe hash algorithm '{algorithm_name}' is not allowed")
    
    # Check if the algorithm is available in hashlib
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
    
    # Generate and return the hash
    hash_obj = hashlib.new(algorithm_name, content)
    return hash_obj.digest()
