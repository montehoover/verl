import re

_LOG_ENTRY_RE = re.compile(r'^\[(?P<timestamp>[^\]]+)\]\s+(?P<log_level>[A-Za-z]+):\s+(?P<message>.*)$')


def _prepare_text(record):
    """
    Normalize and validate the input record.
    Returns a stripped string if valid, otherwise None.
    Pure function: no side effects.
    """
    if not isinstance(record, str):
        return None
    return record.strip()


def _match_log_entry(text):
    """
    Attempt to match the log entry using the predefined regex.
    Returns a match object if matched, otherwise None.
    Pure function: no side effects.
    """
    return _LOG_ENTRY_RE.match(text) if text is not None else None


def _extract_components(match):
    """
    Extract components from a regex match.
    Returns a (timestamp, log_level, message) tuple, or None if match is falsy.
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
    """
    Parse a log entry formatted as: '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    Returns:
      - (timestamp, log_level, message) tuple if properly formatted
      - None otherwise
    Guarantees: No exceptions are raised.
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
