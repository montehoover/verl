import hashlib

def basic_hash(data, algorithm):
    """
    Generate a hash of the input data using the specified algorithm.
    
    Args:
        data: String input to be hashed
        algorithm: Name of the hashing algorithm ('sha256', 'md5', etc.)
    
    Returns:
        String representation of the hash value
    
    Raises:
        ValueError: If an unsupported hash algorithm is specified
    """
    # Convert algorithm name to lowercase for case-insensitive comparison
    algorithm = algorithm.lower()
    
    # Check if the algorithm is supported by hashlib
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Create hash object
    hash_obj = hashlib.new(algorithm)
    
    # Update hash object with the data (encoded to bytes)
    hash_obj.update(data.encode('utf-8'))
    
    # Return the hexadecimal representation of the hash
    return hash_obj.hexdigest()
