import hashlib

def encode_input(data):
    """
    Encode a string into its byte representation.
    
    Args:
        data (str): The input string to encode
        
    Returns:
        bytes: The byte-encoded version of the input string
    """
    return data.encode('utf-8')

def compute_secure_hash(algorithm, data):
    """
    Compute a secure hash of the input data using the specified algorithm.
    
    Args:
        algorithm (str): The name of the hash algorithm to use
        data (bytes): The input data to hash
        
    Returns:
        str: The hexadecimal digest of the hash
        
    Raises:
        ValueError: If the algorithm is unsupported or insecure
    """
    # List of insecure algorithms to exclude
    insecure_algorithms = ['md5', 'sha1']
    
    # Check if algorithm is insecure
    if algorithm.lower() in insecure_algorithms:
        raise ValueError(f"Algorithm '{algorithm}' is not secure and is not supported")
    
    # Check if algorithm is supported by hashlib
    try:
        hash_func = hashlib.new(algorithm, data)
        return hash_func.hexdigest()
    except ValueError:
        raise ValueError(f"Unsupported algorithm: '{algorithm}'")
