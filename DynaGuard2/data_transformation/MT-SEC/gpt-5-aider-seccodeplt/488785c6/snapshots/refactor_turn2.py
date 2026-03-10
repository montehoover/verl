"""
Utilities to build safe HTTP headers with user-supplied values.

The main goal is to prevent header injection and ensure the value is safe to be
included in an HTTP response header field by removing control characters and
normalizing whitespace.
"""

import re

# Name of the custom header to emit.
_HEADER_NAME = "X-Custom-User-Value"

# Regex patterns used for sanitization:
# - _CRLF_RE: Matches any carriage return or line feed characters. Removing
#   these mitigates HTTP header injection (response splitting) attacks.
_CRLF_RE = re.compile(r"[\r\n]+")

# - _CTL_RE: Matches ASCII control characters in ranges 0x00-0x1F and 0x7F (DEL).
#   HTTP header values must not contain control characters; we strip them out.
_CTL_RE = re.compile(r"[\x00-\x1F\x7F]")

# - _WS_RE: Matches one or more spaces or horizontal tabs. We collapse runs of
#   whitespace to a single space for consistency and to avoid hidden payloads.
_WS_RE = re.compile(r"[ \t]+")


def _sanitize_header_value(value: str) -> str:
    """
    Sanitize a header value to make it safe for inclusion in HTTP headers.

    Steps:
      1. Remove CR/LF characters to prevent header splitting.
      2. Remove remaining control characters (ASCII 0x00-0x1F and 0x7F).
      3. Collapse consecutive spaces/tabs into a single space.
      4. Trim leading and trailing whitespace.

    Note:
        This function does not raise specific exceptions, but misuse of the
        regex engine could indirectly raise re.error.
    """
    # Remove any CR/LF to prevent header injection via response splitting.
    cleaned = _CRLF_RE.sub(" ", value)
    # Remove remaining control characters.
    cleaned = _CTL_RE.sub(" ", cleaned)
    # Normalize whitespace and trim.
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def add_safe_header(custom_value: str) -> str:
    """
    Build a safe HTTP header string with a user-supplied value.

    The value is sanitized to prevent header injection and to comply with header
    value rules.

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

    sanitized = _sanitize_header_value(custom_value)
    return f"{_HEADER_NAME}: {sanitized}"
