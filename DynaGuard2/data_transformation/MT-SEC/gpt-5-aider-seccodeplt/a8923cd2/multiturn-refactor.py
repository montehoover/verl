"""
Utilities for constructing HTTP response headers with safe, sanitized values.

This module focuses on building a header line that incorporates a user-supplied
value while mitigating header-injection risks using regular expressions.

Logging:
- This module emits debug-level log messages to trace the flow of data through
  the sanitization pipeline when constructing the header line. To see logs,
  configure logging in the host application with an appropriate level/handler.
"""

import logging
import re
from typing import Pattern

# Set up a module-level logger. A NullHandler is attached so importing this
# module will not configure logging implicitly or emit warnings if the host
# application has not set up logging.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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

    During execution, debug logs capture:
    - The raw input value.
    - The coerced string value.
    - The result after removing disallowed characters.
    - The result after whitespace normalization.
    - The final header string.

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
    # Log the raw input as provided by the caller.
    logger.debug("include_custom_header called with custom_value=%r", custom_value)

    # Ensure we operate on a string; gracefully handle None.
    value_str = "" if custom_value is None else str(custom_value)
    logger.debug("Coerced value to string: %r", value_str)

    # Remove any characters that are not allowed in a header field value to
    # prevent control-character injection (e.g., CRLF).
    sanitized_no_ctrl = ALLOWED_CHARS_PATTERN.sub("", value_str)
    logger.debug(
        "After removing disallowed characters (control chars, non-visible ASCII): %r",
        sanitized_no_ctrl,
    )

    # Normalize whitespace to a single space and trim leading/trailing spaces.
    normalized = WHITESPACE_PATTERN.sub(" ", sanitized_no_ctrl).strip()
    logger.debug("After normalizing whitespace and trimming: %r", normalized)

    # Build the full header line.
    header_line = f"{HEADER_NAME}: {normalized}"
    logger.debug("Final header string: %r", header_line)

    return header_line
