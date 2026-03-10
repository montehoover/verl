import hashlib

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of a given byte sequence using the specified hash algorithm.
    
    Args:
        algorithm_name: The name of the hash algorithm to use
        content: Byte-like object representing the input data
        
    Returns:
        bytes: The generated hash value
        
    Raises:
        ValueError: If the chosen hash algorithm is either unavailable or unsafe to use
    """
    # Define weak/unsafe algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Check if the algorithm is unsafe
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is unsafe and should not be used")
    
    # Check if the algorithm is available in hashlib
    try:
        # Get the hash object for the specified algorithm
        hash_obj = hashlib.new(algorithm_name, content)
        return hash_obj.digest()
    except ValueError:
        # hashlib.new raises ValueError for unsupported algorithms
        raise ValueError(f"Algorithm '{algorithm_name}' is not available")
