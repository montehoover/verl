import re

def append_user_header(user_header: str) -> str:
    # Remove newlines, carriage returns, and control characters
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f-\x9f]', '', user_header)
    return f"Custom-Header: {sanitized_value}"
