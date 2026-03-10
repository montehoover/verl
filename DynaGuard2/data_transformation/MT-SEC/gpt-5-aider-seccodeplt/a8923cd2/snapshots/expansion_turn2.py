def create_header(header_name: str, header_value: str) -> str:
    """
    Construct a raw HTTP header line.

    This function does not validate input; callers that accept user input
    should validate values before calling this helper.
    """
    return f"{header_name}: {header_value}"


def _validate_header_value(value: str) -> None:
    """
    Validate an HTTP header field value per RFC constraints.

    - Must be a string.
    - Must not contain control characters (0x00-0x1F, 0x7F) except HTAB (0x09).
    """
    if not isinstance(value, str):
        raise TypeError("header_value must be a string")

    for ch in value:
        code = ord(ch)
        if code == 0x09:
            continue  # HTAB allowed
        if code < 0x20 or code == 0x7F:
            raise ValueError("Invalid header value: contains control characters")


def add_user_header(header_value: str) -> str:
    """
    Create a 'User-Header' HTTP header line from user-provided input after validation.
    """
    _validate_header_value(header_value)
    return create_header("User-Header", header_value)
