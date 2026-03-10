def create_header(header_name: str, header_value: str) -> str:
    """
    Constructs an HTTP header string.

    Args:
        header_name: The name of the header.
        header_value: The value of the header.

    Returns:
        A string formatted as 'Header-Name: Header-Value'.
    """
    return f"{header_name}: {header_value}"


def add_user_header(header_value: str) -> str:
    """
    Constructs a 'User-Header' string with the given value.

    Args:
        header_value: The value for the User-Header.

    Returns:
        A string formatted as 'User-Header: header_value'.

    Raises:
        ValueError: If the header_value contains invalid characters (e.g., newlines).
    """
    if not isinstance(header_value, str):
        raise TypeError("Header value must be a string.")
    if "\n" in header_value or "\r" in header_value:
        raise ValueError("Header value must not contain newline characters.")
    return f"User-Header: {header_value}"
