import re

def generate_command():
    return 'SELECT * FROM users'

def generate_command_with_header(value):
    return f'SELECT {value} FROM users;'

def include_custom_header(custom_value: str) -> str:
    # Sanitize the custom_value to ensure it's suitable for an HTTP header
    # Remove any control characters and newlines
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f-\x9f]', '', custom_value)
    
    # Create the header string
    return f"X-Custom-Header: {sanitized_value}"
