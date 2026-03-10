import hashlib

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    # Define unsafe algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is unsafe
    if algo_name in unsafe_algorithms:
        raise ValueError(f"Unsafe hashing algorithm '{algo_name}' is not allowed")
    
    # Check if the algorithm is available in hashlib
    try:
        hash_obj = hashlib.new(algo_name)
    except ValueError:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available")
    
    # Update the hash object with the content data
    hash_obj.update(content_data)
    
    # Return the digest as bytes
    return hash_obj.digest()
