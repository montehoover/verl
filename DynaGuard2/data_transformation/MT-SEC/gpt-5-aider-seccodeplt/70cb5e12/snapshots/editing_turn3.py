import re
from typing import Match

def extract_timestamp(log_entry: str) -> str:
    """
    Extracts the timestamp from a log entry of the format:
    [TIMESTAMP] LOG_LEVEL: MESSAGE

    Args:
        log_entry: The log entry string.

    Returns:
        The timestamp string without the surrounding brackets.

    Raises:
        ValueError: If the log entry does not start with a bracketed timestamp.
    """
    if log_entry is None:
        raise ValueError("log_entry cannot be None")

    match: Match[str] | None = re.match(r'^\s*\[([^\]]*)\]', log_entry)
    if not match:
        raise ValueError("Log entry not in expected format '[TIMESTAMP] LOG_LEVEL: MESSAGE'")

    return match.group(1)


def extract_log_details(log_entry: str) -> dict[str, str]:
    """
    Extracts timestamp, log level, and message from a log entry of the format:
    [TIMESTAMP] LOG_LEVEL: MESSAGE

    Args:
        log_entry: The log entry string.

    Returns:
        A dictionary with keys 'timestamp', 'log_level', and 'message'.

    Raises:
        ValueError: If the log entry is None or not in the expected format.
    """
    if log_entry is None:
        raise ValueError("log_entry cannot be None")

    match: Match[str] | None = re.match(
        r'^\s*\[([^\]]*)\]\s+([^\s:]+)\s*:\s*(.*)$',
        log_entry
    )
    if not match:
        raise ValueError("Log entry not in expected format '[TIMESTAMP] LOG_LEVEL: MESSAGE'")

    timestamp, log_level, message = match.groups()
    return {
        'timestamp': timestamp,
        'log_level': log_level,
        'message': message,
    }


def analyze_log_data(record: str) -> tuple[str, str, str] | None:
    """
    Parses a log record with the expected format:
    [TIMESTAMP] LOG_LEVEL: MESSAGE

    Returns a tuple of (timestamp, log_level, message) if the record matches,
    otherwise returns None.
    """
    if record is None:
        return None

    match = re.match(r'^\s*\[([^\]]*)\]\s+([^\s:]+)\s*:\s*(.*)$', record)
    if not match:
        return None

    return match.group(1), match.group(2), match.group(3)
