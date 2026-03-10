import re

def add_custom_header(name: str) -> str:
    return f"print({name})"

def insert_header_with_sanitization(custom_header_value: str) -> str:
    sanitized = re.sub(r'[\x00-\x1F\x7F]', '', custom_header_value)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return f"X-Custom-Header: {sanitized}"
