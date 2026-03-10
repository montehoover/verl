"""
Utilities for constructing a sanitized custom HTTP response header.

This module exposes insert_header_with_sanitization(), which accepts user-
supplied data and returns a header line with the value sanitized to avoid
HTTP header injection and to comply with RFC 7230 field-value requirements.

The function includes debug-level logging to trace:
- The initial input value.
- Each transformation step (disallowed character removal, whitespace collapse,
  trimming, truncation).
- The final sanitized header line.

Note: This module configures a logger but does not set logging handlers or
levels. Integrators should configure logging in the application entrypoint.
"""

import logging
import re

# Module-level logger. Configure handlers/levels in the application as needed.
logger = logging.getLogger(__name__)

# Name of the custom header that will hold the user-provided value.
_HEADER_NAME = "X-Custom-User-Value"

# Hard limit for the header value length to prevent oversized headers.
_MAX_VALUE_LENGTH = 1024

# Maximum number of characters shown in debug logs for any value to
# prevent excessive or potentially sensitive data from flooding logs.
_MAX_LOG_VALUE_LENGTH = 256

# Precompiled regular expressions for performance and readability.

# Matches any character that is NOT:
# - Visible ASCII (0x20 to 0x7E), or
# - obs-text (0x80 to 0xFF) as permitted by RFC 7230.
# This effectively removes control characters (including CR/LF) and any
# Unicode code points above 0xFF.
_DISALLOWED_CHARS_PATTERN = re.compile(r"[^\x20-\x7E\x80-\xFF]+")

# Matches one or more whitespace characters to normalize spacing.
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _repr_for_log(value: object) -> str:
    """
    Return a safe, truncated, repr-like string for logging.

    This prevents unbounded output in logs and makes control characters visible.
    """
    try:
        s = "" if value is None else str(value)
    except Exception:
        # Fallback in case __str__ raises.
        return "<unrepresentable>"

    if len(s) > _MAX_LOG_VALUE_LENGTH:
        s = f"{s[:_MAX_LOG_VALUE_LENGTH]}...(truncated, len={len(s)})"

    return repr(s)


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

    Logging:
        Debug-level logs record the input, each transformation, and the final
        header line to aid troubleshooting.

    Args:
        custom_header_value: The user-provided header value. If None, it is
            treated as an empty string.

    Returns:
        The header line in the form "X-Custom-User-Value: <sanitized>".

    Raises:
        re.error: Only if the underlying regular expressions fail to compile
            or execute (not expected under normal use).
    """
    logger.debug(
        "insert_header_with_sanitization: raw input=%s",
        _repr_for_log(custom_header_value),
    )

    # Ensure value is a string; if None is passed, treat as empty to avoid "None".
    if custom_header_value is None:
        logger.debug("Input is None; treating as empty string.")
        value = ""
    else:
        if not isinstance(custom_header_value, str):
            logger.debug(
                "Input coerced to string via str(): type=%s",
                type(custom_header_value).__name__,
            )
        value = str(custom_header_value)

    # Remove any characters not allowed by RFC 7230 field-value as described
    # above. Disallowed characters are replaced with a single space to keep
    # word boundaries sensible.
    value_no_disallowed, disallowed_replacements = _DISALLOWED_CHARS_PATTERN.subn(
        " ", value
    )
    if disallowed_replacements:
        logger.debug(
            "After removing disallowed characters: replacements=%d, value=%s",
            disallowed_replacements,
            _repr_for_log(value_no_disallowed),
        )
    else:
        logger.debug(
            "No disallowed characters found. Value unchanged: %s",
            _repr_for_log(value_no_disallowed),
        )

    # Collapse all consecutive whitespace into a single space.
    value_ws_collapsed, ws_collapses = _WHITESPACE_PATTERN.subn(
        " ", value_no_disallowed
    )
    if ws_collapses:
        logger.debug(
            "After whitespace collapse: collapses=%d, value=%s",
            ws_collapses,
            _repr_for_log(value_ws_collapsed),
        )
    else:
        logger.debug(
            "No whitespace collapse needed. Value unchanged: %s",
            _repr_for_log(value_ws_collapsed),
        )

    # Trim leading/trailing whitespace.
    trimmed_value = value_ws_collapsed.strip()
    if trimmed_value != value_ws_collapsed:
        trimmed_count = len(value_ws_collapsed) - len(trimmed_value)
        logger.debug(
            "After trim: removed=%d chars, value=%s",
            trimmed_count,
            _repr_for_log(trimmed_value),
        )
    else:
        logger.debug(
            "No trim needed. Value unchanged: %s", _repr_for_log(trimmed_value)
        )

    # Enforce a reasonable maximum length to prevent overly large headers.
    final_value = trimmed_value
    if len(final_value) > _MAX_VALUE_LENGTH:
        original_len = len(final_value)
        final_value = final_value[:_MAX_VALUE_LENGTH]
        logger.debug(
            "After truncation: original_len=%d, truncated_len=%d",
            original_len,
            len(final_value),
        )
    else:
        logger.debug("No truncation applied. Length=%d", len(final_value))

    # Return the final header line.
    header_line = f"{_HEADER_NAME}: {final_value}"
    logger.debug("Final header line: %s", _repr_for_log(header_line))
    return header_line
