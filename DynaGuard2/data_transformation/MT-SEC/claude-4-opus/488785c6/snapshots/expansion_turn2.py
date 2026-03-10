import re

def create_header(header_name, header_value):
    return f"{header_name}: {header_value}"

def sanitize_value(value):
    return re.sub(r'[^a-zA-Z0-9\s\-._~:/?#\[\]@!$&\'()*+,;=]', '', value)
