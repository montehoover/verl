import re

# Precompiled regex patterns for validation and sanitization
_CRLF_RE = re.compile(r"[\r\n]+")
# Control characters except horizontal tab (0x09). CR/LF handled separately above.
_CONTROL_EXCEPT_HTAB_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
# Allowed header value characters: HTAB, space through tilde (0x20-0x7E), and obs-text (0x80-0xFF)
_ALLOWED_VALUE_RE = re.compile(r"^[\t\x20-\x7E\x80-\xFF]*$")
# Inverse of allowed set for sanitization if needed
_NON_ALLOWED_RE = re.compile(r"[^\t\x20-\x7E\x80-\xFF]+")


def append_user_header(user_header: str) -> str:
    """
    Build and return a header string that includes a user-supplied value.

    Args:
        user_header: The custom header value supplied by the user.

    Returns:
        A string in the form 'Custom-Header: <user_header>'.

    Notes:
        - This function does not explicitly raise exceptions, but errors such
          as re.error could occur due to invalid inputs or regex issues.
    """
    # Ensure value is a string (may raise if object's __str__ fails)
    value = user_header if isinstance(user_header, str) else str(user_header)

    # Remove CR/LF to prevent header injection
    value = _CRLF_RE.sub(" ", value)
    # Remove other control characters (except HTAB which is allowed in header values)
    value = _CONTROL_EXCEPT_HTAB_RE.sub("", value)

    # If anything outside the allowed set remains, strip it
    if not _ALLOWED_VALUE_RE.match(value):
        value = _NON_ALLOWED_RE.sub("", value)

    return f"Custom-Header: {value}"
