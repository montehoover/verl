def encode_password(password):
    """
    Convert a password string to its byte representation.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The password encoded as bytes
    """
    return password.encode('utf-8')
