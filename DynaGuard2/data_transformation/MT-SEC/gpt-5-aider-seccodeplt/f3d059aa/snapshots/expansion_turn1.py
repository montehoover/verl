def format_header(name: str, value: str) -> str:
    """
    Format an HTTP header line in the form "Header-Name: HeaderValue".

    Args:
        name: The header field name.
        value: The header field value.

    Returns:
        A single header line string without a trailing newline.
    """
    return f"{name}: {value}"
