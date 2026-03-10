import re

_HEADER_NAME = "X-Custom-User-Value"
_MAX_VALUE_LENGTH = 1024


def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Insert a user-supplied value as a sanitized custom HTTP header line.

    The function sanitizes the input to prevent HTTP header injection and enforces
    a conservative character set in line with RFC 7230:
    - Removes control characters (including CR/LF) and any non-obs-text Unicode.
    - Collapses consecutive whitespace to a single space.
    - Trims leading/trailing whitespace.
    - Limits the value length to a reasonable maximum to avoid oversized headers.

    Args:
        custom_header_value: str - The user-provided custom header value.

    Returns:
        str: The final header string in the format "X-Custom-User-Value: <sanitized>".

    Note:
        This function does not intentionally raise specific exceptions. However, misuse
        of regular expressions in future modifications could indirectly raise re.error.
    """
    # Ensure value is a string; if None is passed, treat as empty string to avoid "None".
    value = "" if custom_header_value is None else str(custom_header_value)

    # Remove any characters not allowed by RFC 7230 field-value:
    # Allow: SP (0x20) to tilde (0x7E) and obs-text (0x80-0xFF).
    # This removes control chars (including CR/LF) and all Unicode above 0xFF.
    value = re.sub(r"[^\x20-\x7E\x80-\xFF]+", " ", value)

    # Collapse all consecutive whitespace into a single space and trim.
    value = re.sub(r"\s+", " ", value).strip()

    # Enforce a reasonable maximum length to prevent overly large headers.
    if len(value) > _MAX_VALUE_LENGTH:
        value = value[:_MAX_VALUE_LENGTH]

    return f"{_HEADER_NAME}: {value}"
