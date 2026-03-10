def create_http_header(header_name: str) -> str:
    """
    Creates a basic HTTP header string.

    Args:
        header_name: The name of the header (e.g., 'Content-Type').

    Returns:
        A string formatted as an HTTP header (e.g., 'Content-Type: default_value').
    """
    # Format the header name: capitalize words and join with hyphens
    formatted_header_name = '-'.join(word.capitalize() for word in header_name.replace('-', ' ').split())
    return f"{formatted_header_name}: default_value"

if __name__ == '__main__':
    # Example usage:
    header1 = create_http_header("Content-Type")
    print(f"'{header1}'")

    header2 = create_http_header("user agent")
    print(f"'{header2}'")

    header3 = create_http_header("X-Custom-Header")
    print(f"'{header3}'")
