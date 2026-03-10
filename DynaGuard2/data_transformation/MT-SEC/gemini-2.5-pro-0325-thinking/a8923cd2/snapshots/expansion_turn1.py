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
