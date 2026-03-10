import re

_HEADER_NAME = "X-Custom-User-Value"
_CRLF_RE = re.compile(r'[\r\n]+')
_CTL_RE = re.compile(r'[\x00-\x1F\x7F]')
_WS_RE = re.compile(r'[ \t]+')


def add_safe_header(custom_value: str) -> str:
    """
    Build a safe HTTP header string with a user-supplied value.

    The value is sanitized to prevent header injection and to comply with header
    value rules:
      - Removes CR/LF and other control characters (0x00-0x1F, 0x7F)
      - Collapses consecutive spaces/tabs into a single space
      - Trims leading/trailing whitespace

    Parameters:
        custom_value (str): The user-provided custom header value.

    Returns:
        str: Header string in the form "X-Custom-User-Value: <sanitized_value>"

    Notes:
        This function does not raise any specific exceptions, but improper input
        could indirectly raise re.error if the regex engine encounters an issue.
    """
    # Coerce non-str input to string to avoid unexpected TypeErrors upstream.
    if not isinstance(custom_value, str):
        custom_value = str(custom_value)

    value = custom_value
    # Remove any CR/LF to prevent header injection via response splitting.
    value = _CRLF_RE.sub(" ", value)
    # Remove remaining control characters.
    value = _CTL_RE.sub(" ", value)
    # Normalize whitespace and trim.
    value = _WS_RE.sub(" ", value).strip()

    return f"{_HEADER_NAME}: {value}"
