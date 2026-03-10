def encode_password(password):
    """
    Encode a password string to bytes.
    
    Args:
        password (str): The password to encode
        
    Returns:
        bytes: The encoded password
    """
    return password.encode('utf-8')
