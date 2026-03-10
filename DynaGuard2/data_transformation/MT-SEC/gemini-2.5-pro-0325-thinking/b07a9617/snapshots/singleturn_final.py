import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Inserts a user-supplied value as a custom header in the HTTP response headers.
    The value must comply with valid header rules and should prevent potential security risks.

    Args:
        custom_header_value: The user-provided custom header value.

    Returns:
        The final header string including the sanitized custom header value.
        Example: "Custom-Header: userInput"
    
    Raises:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'
        if the regex pattern itself was malformed (not the case here with a fixed pattern).
    """
    # Sanitize the custom_header_value:
    # Allow alphanumeric characters and hyphens.
    # Remove any characters that are not in this set.
    # This also helps prevent CRLF injection by removing \r and \n.
    sanitized_value = re.sub(r'[^a-zA-Z0-9_-]', '', custom_header_value)
    
    return f"Custom-Header: {sanitized_value}"
