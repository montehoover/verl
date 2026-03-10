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
