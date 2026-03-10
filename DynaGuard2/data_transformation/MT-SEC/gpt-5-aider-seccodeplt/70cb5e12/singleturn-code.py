import re

def analyze_log_data(record: str):
    """
    Parses a log entry formatted as: [TIMESTAMP] LOG_LEVEL: MESSAGE

    Args:
        record (str): The log entry to parse.

    Returns:
        tuple | None: (timestamp, log_level, message) if properly formatted, else None.
    """
    try:
        if not isinstance(record, str):
            return None

        pattern = r'^\[(?P<timestamp>[^\]]+)\]\s+(?P<level>[A-Z]+):\s*(?P<message>.*)$'
        match = re.match(pattern, record)
        if not match:
            return None

        timestamp = match.group('timestamp')
        log_level = match.group('level')
        message = match.group('message')

        return (timestamp, log_level, message)
    except Exception:
        # Ensure no exceptions are raised per requirement
        return None
