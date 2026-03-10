import re

_HEADER_NAME = "Custom-Header"
# Allow a conservative set of safe ASCII characters for header values.
_ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9 \-._,;:/@()+?=]")

def add_safe_header(custom_value: str) -> str:
    """
    Insert a user-supplied value as a custom header in the HTTP response headers.

    The value is sanitized to comply with valid header rules and to mitigate security risks:
    - Removes CR/LF to prevent header injection (CRLF).
    - Removes other control characters (0x00-0x1F, 0x7F).
    - Normalizes whitespace to single spaces and trims leading/trailing spaces.
    - Removes characters outside a conservative ASCII whitelist.
    - Truncates to a reasonable maximum length.

    Args:
        custom_value: str - the user-provided custom header value.

    Returns:
        str: The final header string including the sanitized custom header value,
             e.g., "Custom-Header: value".

    Note:
        This function does not raise specific exceptions. Improper input could
        indirectly raise 're.error' from the regex engine.
    """
    # Ensure we operate on a string
    if not isinstance(custom_value, str):
        custom_value = str(custom_value)

    # Remove CR and LF to prevent header injection
    value = custom_value.replace("\r", "").replace("\n", "")

    # Remove other control characters (keep only printable range)
    value = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", value)

    # Normalize any whitespace (including tabs and unicode spaces) to single spaces
    value = re.sub(r"\s+", " ", value).strip()

    # Remove characters not in the allowed whitelist
    value = _ALLOWED_CHARS_RE.sub("", value)

    # Enforce a maximum length to avoid excessively large headers
    if len(value) > 1024:
        value = value[:1024]

    return f"{_HEADER_NAME}: {value}"
