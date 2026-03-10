import re

def include_custom_header(custom_value: str) -> str:
    """
    Build a custom HTTP header string with a sanitized value.

    The value is sanitized to remove control characters (except space and tab),
    collapse runs of whitespace, and trim leading/trailing whitespace to mitigate
    header injection risks.

    Args:
        custom_value: User-supplied value for the custom header.

    Returns:
        The full HTTP header line as a string, e.g., 'X-Custom-Header: value'.

    Note:
        This function uses regular expressions for sanitization and does not
        explicitly raise exceptions, but regex-related errors (re.error) could
        propagate if they occur.
    """
    # Ensure input is a string representation
    value_str = "" if custom_value is None else str(custom_value)

    # Remove any characters not allowed in a header field value:
    # Allow horizontal tab (HTAB), space, and visible ASCII 0x21-0x7E.
    # Disallow control chars including CR and LF which can enable injection.
    value_str = re.sub(r'[^\t\x20-\x7E]', '', value_str)

    # Normalize whitespace (spaces and tabs) to a single space
    value_str = re.sub(r'[\t ]+', ' ', value_str).strip()

    header_name = "X-Custom-Header"
    return f"{header_name}: {value_str}"
