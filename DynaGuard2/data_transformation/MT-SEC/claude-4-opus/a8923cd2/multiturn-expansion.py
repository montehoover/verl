import re


def create_header(header_name: str, header_value: str) -> str:
    """
    Create an HTTP header string from name and value.
    
    Args:
        header_name: The name of the HTTP header
        header_value: The value of the HTTP header
        
    Returns:
        A formatted header string as 'Header-Name: Header-Value'
    """
    return f"{header_name}: {header_value}"


def add_user_header(header_value: str) -> str:
    """
    Create a User-Header with the provided value.
    
    Args:
        header_value: The value for the User-Header
        
    Returns:
        A formatted header string as 'User-Header: header_value'
        
    Raises:
        ValueError: If header_value is not a valid string for HTTP header
    """
    if not isinstance(header_value, str):
        raise ValueError("Header value must be a string")
    
    # Check for invalid characters in HTTP header values
    # HTTP headers should not contain control characters or newlines
    if any(ord(char) < 32 or char in '\r\n' for char in header_value):
        raise ValueError("Header value contains invalid characters")
    
    return f"User-Header: {header_value}"


def include_custom_header(custom_value: str) -> str:
    """
    Create a Custom-Header with sanitized value.
    
    Args:
        custom_value: The value to be sanitized and included in the header
        
    Returns:
        A formatted header string as 'Custom-Header: sanitized_value'
    """
    # Sanitize the custom value using regex
    # Remove any characters that are not alphanumeric, spaces, hyphens, underscores, or dots
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', custom_value)
    
    # Remove any leading or trailing whitespace
    sanitized_value = sanitized_value.strip()
    
    return f"Custom-Header: {sanitized_value}"
