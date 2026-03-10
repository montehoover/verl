import re

def sanitize_value(value: str) -> str:
    """
    Sanitize a header value by removing characters that are not
    alphanumeric or basic punctuation.

    Allowed punctuation: space, period, comma, hyphen, underscore, colon,
    semicolon, slash, question mark, exclamation mark, single quote,
    double quote, parentheses, plus, equals.
    """
    return re.sub(r"[^A-Za-z0-9 \.,_\-:;\/\?\!\'\"\(\)\+\=]", "", value)

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
