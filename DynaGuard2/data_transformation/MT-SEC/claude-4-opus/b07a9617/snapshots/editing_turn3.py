import re

def insert_header_with_sanitization(custom_header_value):
    # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', custom_header_value)
    # Remove any line breaks or carriage returns
    sanitized_value = re.sub(r'[\r\n]', '', sanitized_value)
    # Trim whitespace
    sanitized_value = sanitized_value.strip()
    return f'X-Custom-Header: {sanitized_value}'
