import re

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


def validate_header_value(header_value: str) -> bool:
    """
    Validates an HTTP header value against a basic set of rules.

    Checks for presence of newline or carriage return characters, which are
    not allowed in header values.

    Args:
        header_value: The header value string to validate.

    Returns:
        True if the header_value is valid, False otherwise.
    """
    # HTTP header values must not contain CR or LF characters
    if re.search(r"[\r\n]", header_value):
        return False
    return True


def append_user_header(user_header: str) -> str:
    """
    Appends a user-specified header string as a "Custom-Header".

    Validates the user_header value before formatting. If the value is
    invalid (e.g., contains newline characters), it returns an empty string.

    Args:
        user_header: The string value provided by the user for the header.

    Returns:
        A formatted "Custom-Header: <user_header>" string if user_header is valid,
        otherwise an empty string.
    """
    if not validate_header_value(user_header):
        return ""
    return format_header("Custom-Header", user_header)
