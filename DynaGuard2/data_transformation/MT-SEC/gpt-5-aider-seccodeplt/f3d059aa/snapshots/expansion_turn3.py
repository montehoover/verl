import re

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

# Matches sequences of allowed characters for HTTP header field values:
# - HTAB (\x09) and SP (\x20)
# - Visible ASCII characters (\x21-\x7E)
# - obs-text (\x80-\xFF)
_ALLOWED_HEADER_VALUE_RE = re.compile(r'^[\t\x20-\x7E\x80-\xFF]*\Z')

def validate_header_value(header_value: str) -> bool:
    """
    Validate an HTTP header field value against general HTTP specifications.

    This checks that the value does not include prohibited control characters
    such as carriage return (CR) or line feed (LF), and only contains:
      - HTAB and SP
      - visible ASCII (VCHAR)
      - obs-text (0x80-0xFF)

    Args:
        header_value: The header field value to validate.

    Returns:
        True if the value is valid, False otherwise.
    """
    if not isinstance(header_value, str):
        return False
    return _ALLOWED_HEADER_VALUE_RE.match(header_value) is not None

# Disallowed characters for header values: anything not in HTAB, SP, VCHAR, or obs-text.
_DISALLOWED_HEADER_VALUE_RE = re.compile(r'[^\t\x20-\x7E\x80-\xFF]')

def append_user_header(user_header: str) -> str:
    """
    Create a header line for a user-provided value as 'Custom-Header: <user_header>'.

    The value is sanitized to remove characters not permitted in HTTP header values
    (i.e., control characters such as CR/LF). No exceptions are explicitly raised.

    Args:
        user_header: The user-provided header value.

    Returns:
        The formatted header line string.
    """
    if not isinstance(user_header, str):
        try:
            user_value = str(user_header)
        except Exception:
            user_value = ""
    else:
        user_value = user_header

    sanitized_value = _DISALLOWED_HEADER_VALUE_RE.sub("", user_value)
    return format_header("Custom-Header", sanitized_value)
