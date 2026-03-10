import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Insert a user-supplied value as a custom header in the HTTP response headers.
    
    The function sanitizes the input to prevent header injection attacks by removing
    any newline characters and other control characters that could break the HTTP protocol.
    
    Args:
        custom_header_value (str): The user-provided custom header value
        
    Returns:
        str: The final header string including the sanitized custom header value
    """
    # Remove any newline characters, carriage returns, and other control characters
    # to prevent header injection attacks
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f]', '', custom_header_value)
    
    # Return the formatted header string
    return f"Custom-Header: {sanitized_value}"
