import re

def include_custom_header(custom_value: str) -> str:
    """
    Inserts a custom header into the response headers with a user-supplied value.
    
    Args:
        custom_value: The user-defined value for the custom header
        
    Returns:
        The full header string in the format "Custom-Header: value"
    """
    # Sanitize the custom value to ensure it's valid for HTTP headers
    # Remove any newline characters or carriage returns that could cause header injection
    sanitized_value = re.sub(r'[\r\n]', '', str(custom_value))
    
    # Optionally, you might want to further sanitize by removing or escaping other problematic characters
    # For now, we'll just ensure no newlines can break the header structure
    
    return f"Custom-Header: {sanitized_value}"
