import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    # Remove any characters that are not alphanumeric, spaces, hyphens, underscores, or common punctuation
    # This prevents header injection attacks by removing newlines, carriage returns, and other control characters
    sanitized_value = re.sub(r'[^\w\s\-_.,;:!?()]', '', custom_header_value)
    
    # Remove any leading or trailing whitespace
    sanitized_value = sanitized_value.strip()
    
    # Limit the length to prevent excessively long headers
    max_length = 8192  # Common max header value length
    if len(sanitized_value) > max_length:
        sanitized_value = sanitized_value[:max_length]
    
    # Create the final header string
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    return header_string
