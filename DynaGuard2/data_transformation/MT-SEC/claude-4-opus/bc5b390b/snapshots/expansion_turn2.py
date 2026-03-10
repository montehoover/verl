import hashlib

def basic_hash(input_string):
    """
    Takes a string input and returns its SHA-256 hash.
    
    Args:
        input_string (str): The string to hash
        
    Returns:
        str: The hexadecimal representation of the SHA-256 hash
    """
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the bytes of the input string
    hash_object.update(input_string.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()

def validate_algorithm(algorithm_name):
    """
    Validates if a hash algorithm is secure.
    
    Args:
        algorithm_name (str): The name of the hash algorithm to validate
        
    Returns:
        bool: True if the algorithm is secure
        
    Raises:
        ValueError: If the algorithm is not secure
    """
    # Define secure algorithms
    secure_algorithms = {
        'sha256', 'sha384', 'sha512', 
        'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    # Define insecure algorithms
    insecure_algorithms = {
        'md5', 'sha1', 'sha', 'md4', 'md2'
    }
    
    # Normalize the algorithm name to lowercase
    algorithm_lower = algorithm_name.lower()
    
    # Check if algorithm is secure
    if algorithm_lower in secure_algorithms:
        return True
    elif algorithm_lower in insecure_algorithms:
        raise ValueError(f"Algorithm '{algorithm_name}' is not secure and should not be used")
    else:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'")
