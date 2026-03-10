import re

def add_header(header_value):
    if all(c.isalnum() or c.isspace() for c in header_value):
        return f'Custom-Header: {header_value}'
    else:
        return 'Error: Header value contains invalid characters'

def add_safe_header(custom_value):
    # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
    # This prevents header injection attacks by removing newlines, carriage returns, etc.
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', custom_value)
    
    # Remove any leading/trailing whitespace
    sanitized_value = sanitized_value.strip()
    
    # If the value is empty after sanitization, return a default header
    if not sanitized_value:
        return 'X-Custom-Header: default'
    
    # Return the sanitized header
    return f'X-Custom-Header: {sanitized_value}'
