import re

def create_header(header_name, header_value):
    return f"{header_name}: {header_value}"

def sanitize_value(value):
    return re.sub(r'[^a-zA-Z0-9\s\-._~:/?#\[\]@!$&\'()*+,;=]', '', value)

def add_safe_header(custom_value):
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-._~:/?#\[\]@!$&\'()*+,;=]', '', custom_value)
    return f"Custom-Header: {sanitized_value}"
