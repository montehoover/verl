import re

_LOG_ENTRY_RE = re.compile(
    r'^\s*\[(?P<timestamp>.*?)\]\s+(?P<log_level>[A-Za-z]+)\s*:\s*(?P<message>.*)\s*$'
)


def _match_log_entry(log_line: str):
    """Return a regex match for a log entry or None.

    Attempt to match the supplied log line against the expected pattern
    '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_line: The log entry string to parse.

    Returns:
        A re.Match object if the string matches the log pattern; otherwise None.

    This function swallows all exceptions and returns None.
    """
    try:
        if not isinstance(log_line, str):
            return None
        return _LOG_ENTRY_RE.match(log_line)
    except Exception:
        return None


def extract_log_data(log_line: str):
    """Extract timestamp, log level, and message from a log line.

    Parses entries in the format '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_line: The log entry string to parse.

    Returns:
        A tuple of (timestamp, log_level, message) on success; otherwise None.

    This function avoids raising exceptions and returns None on errors.
    """
    try:
        if not isinstance(log_line, str):
            return None

        match = _match_log_entry(log_line)
        if not match:
            return None

        return (
            match.group('timestamp'),
            match.group('log_level'),
            match.group('message'),
        )
    except Exception:
        return None
