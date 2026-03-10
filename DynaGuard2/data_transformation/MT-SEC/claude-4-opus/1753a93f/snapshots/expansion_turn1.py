import hashlib

def compute_hash(data, algorithm):
    """
    Compute the hash of input data using the specified algorithm.
    
    Args:
        data: String input to hash
        algorithm: Hash algorithm name (e.g., 'sha256', 'sha512', 'md5')
    
    Returns:
        Hexadecimal string representation of the hash
    
    Raises:
        ValueError: If the algorithm is not supported
    """
    # Convert algorithm name to lowercase for case-insensitive matching
    algorithm = algorithm.lower()
    
    # Check if the algorithm is available in hashlib
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Create hash object
    hash_obj = hashlib.new(algorithm)
    
    # Update hash with encoded data
    hash_obj.update(data.encode('utf-8'))
    
    # Return hexadecimal digest
    return hash_obj.hexdigest()
