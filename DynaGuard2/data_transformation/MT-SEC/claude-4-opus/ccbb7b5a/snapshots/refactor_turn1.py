import hashlib

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    # Define weak/unsafe algorithms
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if algorithm is unsafe
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is unsafe and should not be used")
    
    # Check if algorithm is available
    try:
        hash_obj = hashlib.new(algorithm_name, content)
        return hash_obj.digest()
    except ValueError:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available")
