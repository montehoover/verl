"""
Utilities for constructing HTTP response headers with safe, sanitized values.

This module focuses on building a header line that incorporates a user-supplied
value while mitigating header-injection risks using regular expressions.
"""

import re
from typing import Pattern

# Constants
HEADER_NAME: str = "X-Custom-Header"

# Precompiled regular expressions for performance and maintainability.
# ALLOWED_CHARS_PATTERN removes any character that is not:
# - Horizontal tab (HTAB, \t),
# - Space (SP, 0x20),
# - Visible ASCII characters from 0x21 ('!') to 0x7E ('~').
# This excludes control characters such as carriage return (CR) and line feed
# (LF), which could otherwise be abused for header injection.
ALLOWED_CHARS_PATTERN: Pattern[str] = re.compile(r"[^\t\x20-\x7E]")

# WHITESPACE_PATTERN collapses consecutive spaces and tabs into a single space.
WHITESPACE_PATTERN: Pattern[str] = re.compile(r"[\t ]+")


def include_custom_header(custom_value: str) -> str:
    """
    Return a complete HTTP header line with a sanitized, user-supplied value.

    Sanitization steps:
    - Coerce the input to a string (None becomes an empty string).
    - Remove disallowed characters (anything other than HTAB, SP, and visible
      ASCII 0x21–0x7E).
    - Collapse runs of spaces and tabs to a single space, then trim the result.

    Parameters
    ----------
    custom_value : str
        User-defined value for the custom header.

    Returns
    -------
    str
        The full HTTP header line in the form:
        'X-Custom-Header: <sanitized-value>'.

    Notes
    -----
    This function does not explicitly raise exceptions, but regular-expression
    compilation or substitution errors may propagate as 're.error'.
    """
    # Ensure we operate on a string; gracefully handle None.
    value_str = "" if custom_value is None else str(custom_value)

    # Remove any characters that are not allowed in a header field value to
    # prevent control-character injection (e.g., CRLF).
    value_str = ALLOWED_CHARS_PATTERN.sub("", value_str)

    # Normalize whitespace to a single space and trim leading/trailing spaces.
    value_str = WHITESPACE_PATTERN.sub(" ", value_str).strip()

    # Build and return the full header line.
    return f"{HEADER_NAME}: {value_str}"
