"""Utilities for parsing log entries.

This module provides a robust, readable, and PEP-8-compliant implementation
for parsing log entries formatted as: '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

The main entry point is `analyze_log_data`, which extracts the timestamp,
log level, and message from a single log record. The function is designed
to be safe and never raise exceptions; it returns None for invalid inputs.
"""

import re


_LOG_ENTRY_RE = re.compile(
    r'^\[(?P<timestamp>[^\]]+)\]\s+'
    r'(?P<log_level>[A-Za-z]+):\s+'
    r'(?P<message>.*)$'
)


def _prepare_text(record):
    """Normalize and validate the input record.

    This function ensures the input is a string and trims leading and trailing
    whitespace.

    Args:
        record: The input log record. Expected to be a string.

    Returns:
        A stripped string if the input is a valid string; otherwise, None.

    Notes:
        Pure function: no side effects.
    """
    if not isinstance(record, str):
        return None
    return record.strip()


def _match_log_entry(text):
    """Attempt to match a log entry against the predefined regex.

    Args:
        text: A normalized string (ideally produced by `_prepare_text`).

    Returns:
        A regex match object if the text matches the expected format;
        otherwise, None.

    Notes:
        Pure function: no side effects.
    """
    return _LOG_ENTRY_RE.match(text) if text is not None else None


def _extract_components(match):
    """Extract components from a regex match.

    Args:
        match: A match object produced by `_match_log_entry`.

    Returns:
        A tuple (timestamp, log_level, message) if `match` is truthy;
        otherwise, None.

    Notes:
        Pure function: no side effects.
    """
    if not match:
        return None

    return (
        match.group('timestamp'),
        match.group('log_level'),
        match.group('message'),
    )


def analyze_log_data(record: str):
    """Parse a log entry formatted as: '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        record: The raw log entry string to parse.

    Returns:
        A tuple (timestamp, log_level, message) if the log entry is properly
        formatted; otherwise, None.

    Guarantees:
        - No exceptions are raised (returns None on any error).
    """
    try:
        text = _prepare_text(record)
        if text is None:
            return None

        match = _match_log_entry(text)
        if not match:
            return None

        return _extract_components(match)
    except Exception:
        return None
