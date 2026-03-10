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
