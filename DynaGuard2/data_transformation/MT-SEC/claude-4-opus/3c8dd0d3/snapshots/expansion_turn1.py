def encode_password(password):
    """
    Encode a password string to bytes for secure storage.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The password encoded as bytes
    """
    return password.encode('utf-8')
