import re

def include_custom_header(custom_value: str) -> str:
    """
    Inserts a custom HTML header into the response headers.

    The user input must be a valid string suitable for an HTTP header field.

    Args:
        custom_value: str, serving as the user-defined value for the custom header.

    Returns:
        str, representing the full header string which incorporates the sanitized
        custom header value.

    Raises:
        While no explicit exceptions are raised, improper input or regex-related
        problems could cause 're.error' exceptions.
    """
    # Sanitize the custom_value to ensure it's a valid HTTP header field value.
    # HTTP header field values should ideally be restricted to visible ASCII characters.
    # For this example, we'll allow alphanumeric characters, hyphens, and underscores.
    # Other characters will be removed.
    sanitized_value = re.sub(r'[^a-zA-Z0-9_-]', '', custom_value)
    return f'X-Custom-Header: {sanitized_value}'
