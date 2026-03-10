import re

def append_user_header(user_header: str) -> str:
    """
    Inserts a custom header into the output headers of a response.

    Args:
        user_header: str, representing the custom header value input by the user.

    Returns:
        str, which contains the full header string including the custom header value.

    Raises:
        TypeError: If user_header is not a string.
        ValueError: If user_header is empty or contains invalid characters (newline, carriage return).
    """
    if not isinstance(user_header, str):
        raise TypeError("Header value must be a string.")

    if not user_header:
        raise ValueError("Header value cannot be empty.")

    if re.search(r"[\n\r]", user_header):
        raise ValueError("Header value cannot contain newline or carriage return characters.")

    header_name = "X-User-Header"
    return f"{header_name}: {user_header}"
