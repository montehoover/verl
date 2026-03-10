import hashlib

def validate_algorithm(algo_name: str) -> None:
    """
    Validate that the given algorithm name is supported by hashlib.
    
    Args:
        algo_name: The name of the cryptographic algorithm
        
    Raises:
        ValueError: If the algorithm is not supported
    """
    try:
        hashlib.new(algo_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")

def compute_hash(algo_name: str, data: bytes) -> bytes:
    """
    Compute the hash of the given data using the specified algorithm.
    
    Args:
        algo_name: The name of the cryptographic algorithm
        data: The data to be hashed
        
    Returns:
        The computed hash value
    """
    hash_obj = hashlib.new(algo_name)
    hash_obj.update(data)
    return hash_obj.digest()

def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using a specified secure algorithm.
    
    Args:
        algo_name: The name of the cryptographic algorithm to utilize for hashing
        pass_key: The password to be hashed
        
    Returns:
        The computed hash value of the specified password
        
    Raises:
        ValueError: If the given hash algorithm is not found in the list of supported algorithms
    """
    validate_algorithm(algo_name)
    return compute_hash(algo_name, pass_key)
