import hashlib

def compute_digest(algorithm_name, content):
    """Compute hash of content using specified algorithm and return as bytes."""
    # Define weak algorithms that should not be used
    weak_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if algorithm is weak
    if algorithm_name in weak_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is unsafe and should not be used")
    
    # Check if algorithm is available
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not available in hashlib")
    
    # Compute and return hash as bytes
    hash_object = hashlib.new(algorithm_name, content)
    return hash_object.digest()
