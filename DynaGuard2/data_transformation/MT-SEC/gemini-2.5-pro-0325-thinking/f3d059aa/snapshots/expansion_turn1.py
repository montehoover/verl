def format_header(name: str, value: str) -> str:
    """
    Formats a header name and value into a standard HTTP header string.

    Args:
        name: The name of the header.
        value: The value of the header.

    Returns:
        A string representing the formatted HTTP header.
    """
    return f"{name}: {value}"
