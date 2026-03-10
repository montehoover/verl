import re

def create_http_header(header_name):
    """
    Create an HTTP header string from a header name.
    
    Args:
        header_name (str): The name of the HTTP header
        
    Returns:
        str: Formatted HTTP header string
    """
    # Convert header_name to proper HTTP header format
    # Split by underscore or hyphen, capitalize each word, join with hyphen
    parts = header_name.replace('_', '-').split('-')
    formatted_name = '-'.join(word.capitalize() for word in parts)
    
    return f"{formatted_name}: value"

def sanitize_input(user_input):
    """
    Sanitize user input for use in HTTP headers.
    
    Args:
        user_input (str): The user input string to sanitize
        
    Returns:
        str: Sanitized string safe for HTTP headers
    """
    # Remove any characters that aren't allowed in HTTP headers
    # Allow only alphanumeric, spaces, hyphens, underscores, and common header-safe characters
    sanitized = re.sub(r'[^\w\s\-_.,;:!@#$%&*()+=\[\]{}|\\/<>?~`]', '', user_input)
    
    # Remove any control characters and non-printable characters
    sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', sanitized)
    
    # Remove any line breaks or carriage returns
    sanitized = re.sub(r'[\r\n]', '', sanitized)
    
    return sanitized

def insert_header_with_sanitization(custom_header_value):
    """
    Insert a user-supplied value as a custom header with sanitization.
    
    Args:
        custom_header_value (str): The custom header value to insert
        
    Returns:
        str: The final header string with sanitized custom header value
    """
    # Sanitize the custom header value
    sanitized_value = sanitize_input(custom_header_value)
    
    # Create the custom header string
    return f"X-Custom-Header: {sanitized_value}"
