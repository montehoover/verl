"""
Utilities for constructing a sanitized custom HTTP response header.

This module exposes insert_header_with_sanitization(), which accepts user-
supplied data and returns a header line with the value sanitized to avoid
HTTP header injection and to comply with RFC 7230 field-value requirements.
"""

import re

# Name of the custom header that will hold the user-provided value.
_HEADER_NAME = "X-Custom-User-Value"

# Hard limit for the header value length to prevent oversized headers.
_MAX_VALUE_LENGTH = 1024

# Precompiled regular expressions for performance and readability.

# Matches any character that is NOT:
# - Visible ASCII (0x20 to 0x7E), or
# - obs-text (0x80 to 0xFF) as permitted by RFC 7230.
# This effectively removes control characters (including CR/LF) and any
# Unicode code points above 0xFF.
_DISALLOWED_CHARS_PATTERN = re.compile(r"[^\x20-\x7E\x80-\xFF]+")

# Matches one or more whitespace characters to normalize spacing.
_WHITESPACE_PATTERN = re.compile(r"\s+")


def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Build a header line containing a sanitized, user-provided value.

    The value is normalized to a conservative character set to mitigate header
    injection and to conform to RFC 7230 field-value constraints.

    Sanitization steps:
      - Remove control characters (including CR and LF) and any character
        outside ASCII visible (0x20-0x7E) and obs-text (0x80-0xFF).
      - Collapse consecutive whitespace to a single space.
      - Trim leading/trailing whitespace.
      - Truncate to at most 1024 characters.

    Args:
        custom_header_value: The user-provided header value. If None, it is
            treated as an empty string.

    Returns:
        The header line in the form "X-Custom-User-Value: <sanitized>".

    Raises:
        re.error: Only if the underlying regular expressions fail to compile
            or execute (not expected under normal use).
    """
    # Ensure value is a string; if None is passed, treat as empty to avoid "None".
    value = "" if custom_header_value is None else str(custom_header_value)

    # Remove any characters not allowed by RFC 7230 field-value as described
    # above. Disallowed characters are replaced with a single space to keep
    # word boundaries sensible.
    value = _DISALLOWED_CHARS_PATTERN.sub(" ", value)

    # Collapse all consecutive whitespace into a single space and trim.
    value = _WHITESPACE_PATTERN.sub(" ", value).strip()

    # Enforce a reasonable maximum length to prevent overly large headers.
    if len(value) > _MAX_VALUE_LENGTH:
        value = value[:_MAX_VALUE_LENGTH]

    # Return the final header line.
    return f"{_HEADER_NAME}: {value}"
