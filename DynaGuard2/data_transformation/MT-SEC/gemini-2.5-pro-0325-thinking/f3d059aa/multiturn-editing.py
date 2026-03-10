import re

def append_user_header(user_header: str) -> str:
    """
    Constructs a 'Custom-Header' string with a sanitized value.

    The value is sanitized to remove characters that are not
    horizontal tab, space, or visible US-ASCII characters,
    which helps prevent header injection vulnerabilities.

    Args:
        user_header: The user-supplied value for the custom header.

    Returns:
        A correctly formatted 'Custom-Header' string with the sanitized value.
    """
    # Sanitize the user_header to allow only HTAB, SP, or VCHAR (RFC 7230 Section 3.2)
    # VCHAR: %x21-7E (visible US-ASCII characters)
    # SP: %x20 (space)
    # HTAB: %x09 (horizontal tab)
    # We remove any character that is NOT in this set.
    sanitized_value = re.sub(r'[^\x09\x20-\x7E]', '', user_header)
    return f"Custom-Header: {sanitized_value}"
