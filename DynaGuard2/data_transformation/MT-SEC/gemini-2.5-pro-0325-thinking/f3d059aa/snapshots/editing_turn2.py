import re

def append_user_header(user_header: str) -> str:
    """
    Constructs an 'X-Custom-Header' string with a sanitized value.

    Args:
        user_header: The value for the custom header.

    Returns:
        A correctly formatted 'X-Custom-Header' string.
    """
    # Sanitize the user_header to include only alphanumeric characters, dashes, and underscores
    sanitized_header_value = re.sub(r'[^a-zA-Z0-9_-]', '', user_header)
    return f"X-Custom-Header: {sanitized_header_value}"
