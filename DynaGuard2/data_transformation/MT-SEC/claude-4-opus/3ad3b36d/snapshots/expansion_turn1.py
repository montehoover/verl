def encode_password(password):
    """
    Convert a plaintext password to its byte representation.
    
    Args:
        password (str): The plaintext password to encode
        
    Returns:
        bytes: The UTF-8 encoded byte representation of the password
    """
    return password.encode('utf-8')
