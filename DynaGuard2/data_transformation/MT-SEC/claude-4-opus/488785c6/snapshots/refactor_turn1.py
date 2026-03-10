import re

def add_safe_header(custom_value: str) -> str:
    # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
    # This prevents header injection attacks by removing newlines, carriage returns, and other control characters
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', custom_value)
    
    # Remove any leading or trailing whitespace
    sanitized_value = sanitized_value.strip()
    
    # Limit the length to prevent excessively long headers
    sanitized_value = sanitized_value[:100]
    
    # Create the header string
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    return header_string
