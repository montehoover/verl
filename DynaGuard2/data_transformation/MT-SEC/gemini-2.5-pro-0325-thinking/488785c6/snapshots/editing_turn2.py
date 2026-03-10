import re

def add_header(header_value: str) -> str:
    """
    Adds a simple custom header to an HTTP response.
    Validates that the header_value contains only alphanumeric characters and spaces.

    Args:
        header_value: The value for the custom header.

    Returns:
        A string formatted as 'Custom-Header: {header_value}' if valid,
        otherwise an error message.
    """
    if re.match("^[a-zA-Z0-9 ]*$", header_value):
        return f"Custom-Header: {header_value}"
    else:
        return "Error: Header value must contain only alphanumeric characters and spaces."
