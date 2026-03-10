def add_header(header_value: str) -> str:
    """
    Adds a simple custom header to an HTTP response.

    Args:
        header_value: The value for the custom header.

    Returns:
        A string formatted as 'Custom-Header: {header_value}'.
    """
    return f"Custom-Header: {header_value}"
