import re

def add_safe_header(custom_value: str) -> str:
    """
    Inserts a user-supplied value as a custom header in the HTTP response headers.
    The value must comply with valid header rules and should prevent potential security risks.

    Args:
        custom_value: str, the user-provided custom header value.

    Returns:
        str, representing the final header string including the sanitized custom header value.
    
    Raises:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'.
    """
    # Remove any characters not allowed in a header value.
    # Valid characters are typically alphanumeric and some special characters like -.
    # We explicitly disallow newline or carriage return characters to prevent header injection.
    sanitized_value = re.sub(r'[^\w\s-]', '', custom_value)
    sanitized_value = re.sub(r'[\r\n]', '', sanitized_value) # Ensure no newlines
    
    # Construct the header string
    header_name = "X-Custom-Header"
    return f"{header_name}: {sanitized_value}"
