import hashlib

def basic_hash(data, algorithm='sha256'):
    """
    Generate a hash value for the given data using the specified algorithm.
    
    Args:
        data (str): The input string to be hashed
        algorithm (str): The hash algorithm to use (e.g., 'sha256', 'md5', 'sha1')
    
    Returns:
        str: The hexadecimal hash value
    """
    # Get the hash algorithm from hashlib
    hash_func = getattr(hashlib, algorithm)()
    
    # Update the hash with the data (encoded to bytes)
    hash_func.update(data.encode('utf-8'))
    
    # Return the hexadecimal digest
    return hash_func.hexdigest()
