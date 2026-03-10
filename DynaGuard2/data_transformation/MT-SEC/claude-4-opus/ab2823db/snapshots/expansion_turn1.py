def encode_password(password):
    """
    Takes a password as a string and returns its byte representation.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The UTF-8 encoded byte representation of the password
    """
    return password.encode('utf-8')
