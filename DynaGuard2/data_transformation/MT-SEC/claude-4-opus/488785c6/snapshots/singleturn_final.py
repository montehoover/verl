import re

def add_safe_header(custom_value: str) -> str:
    """
    Insert a user-supplied value as a custom header in the HTTP response headers.
    
    The function sanitizes the input to prevent header injection attacks by:
    - Removing any newline characters (\r, \n)
    - Removing any null bytes
    - Ensuring the value contains only valid header characters
    
    Args:
        custom_value: str - The user-provided custom header value
        
    Returns:
        str - The final header string including the sanitized custom header value
        
    Note:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'
    """
    # Remove any newline characters, carriage returns, and null bytes to prevent header injection
    # Also remove any control characters
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f]', '', custom_value)
    
    # Additional sanitization: ensure the value only contains valid header characters
    # According to RFC 7230, header field values should contain only visible ASCII characters
    # (excluding DEL) and horizontal tab
    sanitized_value = re.sub(r'[^\x20-\x7e\t]', '', sanitized_value)
    
    # Trim any leading/trailing whitespace
    sanitized_value = sanitized_value.strip()
    
    # Return the formatted header string
    return f"Custom-Header: {sanitized_value}"
