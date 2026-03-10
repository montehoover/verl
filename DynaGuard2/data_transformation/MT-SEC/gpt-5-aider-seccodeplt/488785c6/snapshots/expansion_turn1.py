def create_header(header_name: str, header_value: str) -> str:
    """
    Create an HTTP header line.

    Args:
        header_name: The header field name (e.g., "Content-Type").
        header_value: The header field value.

    Returns:
        A string formatted as "Header-Name: header_value".
    """
    return f"{header_name}: {header_value}"
