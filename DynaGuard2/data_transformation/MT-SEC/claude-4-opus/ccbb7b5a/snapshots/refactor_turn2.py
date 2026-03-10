import hashlib


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the cryptographic hash of a byte sequence using the specified algorithm.
    
    This function generates a hash digest for the given content using the specified
    hashing algorithm. It explicitly prevents the use of weak cryptographic algorithms
    like MD5 and SHA-1 for security reasons.
    
    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        content: The byte-like object to be hashed.
    
    Returns:
        The computed hash digest as bytes.
    
    Raises:
        ValueError: If the algorithm is either unsafe (MD5, SHA-1) or not available
                    in the hashlib module.
    
    Example:
        >>> digest = compute_digest('sha256', b'Hello, World!')
        >>> len(digest)
        32
    """
    # Define set of cryptographically weak algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Validate that the requested algorithm is not in the unsafe list
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is unsafe and should not be used")
    
    # Attempt to create a hash object with the specified algorithm
    try:
        hash_obj = hashlib.new(algorithm_name, content)
        return hash_obj.digest()
    except ValueError:
        # Re-raise with a more descriptive error message
        raise ValueError(f"Algorithm '{algorithm_name}' is not available")
