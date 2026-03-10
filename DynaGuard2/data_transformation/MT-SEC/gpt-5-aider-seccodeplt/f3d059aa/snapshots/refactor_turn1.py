import re

# Precompiled regex patterns for sanitizing header values
_CRLF_NUL_RE = re.compile(r'[\r\n\x00]')
_INVALID_CTL_RE = re.compile(r'[\x00-\x08\x0A-\x1F\x7F]')


def append_user_header(user_header: str) -> str:
    """
    Insert a custom user header into the response headers.

    The function takes a user-provided header value, sanitizes it to ensure it is
    safe for use in an HTTP header (removing control characters and CR/LF to
    prevent header injection), and returns the full header line with a CRLF
    terminator.

    Does not explicitly raise exceptions; however, regex-related errors could
    occur if the environment's regex engine fails unexpectedly.

    Args:
        user_header: The custom header value input by the user.

    Returns:
        The full HTTP header string including the custom header value, terminated
        by CRLF.
    """
    if not isinstance(user_header, str):
        try:
            user_header = "" if user_header is None else str(user_header)
        except Exception:
            user_header = ""

    # Trim outer whitespace
    value = user_header.strip()

    # Remove CR, LF, and NUL to prevent header injection
    value = _CRLF_NUL_RE.sub("", value)

    # Remove other control characters (except horizontal tab which is normalized below)
    value = _INVALID_CTL_RE.sub("", value)

    # Normalize runs of spaces/tabs to a single space
    value = re.sub(r"[ \t]+", " ", value)

    # Build the header line with standard CRLF terminator
    return f"X-User-Header: {value}\r\n"
