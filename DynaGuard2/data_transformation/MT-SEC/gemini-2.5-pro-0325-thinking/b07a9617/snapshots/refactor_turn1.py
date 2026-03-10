import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Inserts a user-supplied value as a custom header in the HTTP response headers.
    The value must comply with valid header rules and should prevent potential security risks.

    Args:
        custom_header_value: str, the user-provided custom header value.

    Returns:
        str, representing the final header string including the sanitized custom header value.
    
    Raises:
        Does not raise any specific exceptions, but improper input could indirectly raise 're.error'.
    """
    # Sanitize the header value:
    # Allow printable ASCII characters (decimal 32-126) except for characters
    # that might be problematic in headers (e.g., colon, semicolon, backslash, quotes, etc.
    # depending on strictness). For a general approach, we'll remove non-printable ASCII
    # and characters like \r, \n to prevent header injection.
    # A more restrictive approach might be [a-zA-Z0-9_.-]
    # For this example, we'll remove anything that's not a letter, digit, or common safe punctuation.
    # We also explicitly remove \r and \n to prevent HTTP response splitting.
    sanitized_value = re.sub(r'[\r\n]', '', custom_header_value) # Remove CR and LF
    # Further restrict to a safe set of characters for header values
    # This regex allows alphanumeric characters, spaces, and some common punctuation.
    # It disallows characters like ':', ';', '<', '>', '"', '\'', '\\' which can be problematic.
    sanitized_value = re.sub(r'[^a-zA-Z0-9_ .!#$%&\'*+\-/^`|~]+', '', sanitized_value)
    
    # HTTP headers should not ideally start or end with whitespace, though many servers trim it.
    sanitized_value = sanitized_value.strip()

    return f"X-Custom-Header: {sanitized_value}"
