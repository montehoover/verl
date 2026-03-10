import re

def include_custom_header(custom_value: str) -> str:
    """
    Inserts a custom header into the response headers, where the header value is supplied by the user.

    The user input must be a valid string suitable for an HTTP header field.

    Args:
        custom_value: str, serving as the user-defined value for the custom header.

    Returns:
        str, representing the full header string which incorporates the sanitized custom header value.

    Raises:
        re.error: Can be raised if there are issues with the regular expression pattern
                  (though unlikely with a fixed pattern).
    """
    # Sanitize the custom_value to allow only printable ASCII characters (space through tilde)
    # This helps prevent header injection and ensures the value is suitable for an HTTP header.
    sanitized_value = re.sub(r'[^\x20-\x7E]', '', custom_value)
    
    return f"Custom-Header: {sanitized_value}"
