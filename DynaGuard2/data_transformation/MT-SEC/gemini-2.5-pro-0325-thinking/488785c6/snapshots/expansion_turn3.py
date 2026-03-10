import re

def create_header(header_name: str, header_value: str) -> str:
    """
    Creates a formatted HTTP header string.

    Args:
        header_name: The name of the header.
        header_value: The value of the header.

    Returns:
        A string formatted as "Header-Name: header_value".
    """
    return f"{header_name}: {header_value}"


def sanitize_value(value: str) -> str:
    """
    Sanitizes a string by removing characters that are not
    alphanumeric or basic punctuation (space, period, hyphen, underscore).

    Args:
        value: The string to sanitize.

    Returns:
        The sanitized string.
    """
    # Keep alphanumeric characters, spaces, periods, hyphens, and underscores
    return re.sub(r'[^a-zA-Z0-9 ._-]', '', value)


def add_safe_header(custom_value: str) -> str:
    """
    Creates a formatted HTTP header string for "Custom-Header"
    with a sanitized value.

    Args:
        custom_value: The value for the "Custom-Header".

    Returns:
        A string formatted as "Custom-Header: sanitized_value".
    """
    sanitized_custom_value = sanitize_value(custom_value)
    return create_header("Custom-Header", sanitized_custom_value)
