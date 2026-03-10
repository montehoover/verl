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
