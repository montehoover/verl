import re

def include_custom_header(custom_value: str) -> str:
    # Sanitize the custom value to ensure it's valid for HTTP headers
    # Remove any control characters and newlines
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f]', '', custom_value)
    
    # Create the full header string with a custom header name
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    return header_string
