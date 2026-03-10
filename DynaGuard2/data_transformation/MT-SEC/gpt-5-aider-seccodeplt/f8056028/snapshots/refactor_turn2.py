import re

_LOG_ENTRY_RE = re.compile(
    r'^\s*\[(?P<timestamp>.*?)\]\s+(?P<log_level>[A-Za-z]+)\s*:\s*(?P<message>.*)\s*$'
)


def _match_log_entry(log_line: str):
    """
    Attempt to match a log entry against the expected pattern.

    Returns a re.Match object if the line matches; otherwise, returns None.
    The function avoids raising exceptions.
    """
    try:
        if not isinstance(log_line, str):
            return None
        return _LOG_ENTRY_RE.match(log_line)
    except Exception:
        return None


def extract_log_data(log_line: str):
    """
    Extract (timestamp, log_level, message) from a log line.

    Returns a tuple on success, or None on failure. The function does not raise
    exceptions.
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
