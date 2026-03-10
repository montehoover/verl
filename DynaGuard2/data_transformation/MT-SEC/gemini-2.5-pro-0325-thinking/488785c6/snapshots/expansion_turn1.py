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
