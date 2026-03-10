import re

def analyze_log_data(record: str):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.

    Args:
        record: str, the log entry that needs parsing.

    Returns:
        A tuple (timestamp, log_level, message) if the log entry is properly formatted.
        Otherwise, return None.
    """
    log_pattern = re.compile(r"\[(.*?)\]\s(.*?):\s(.*)")
    match = log_pattern.match(record)
    if match:
        timestamp, log_level, message = match.groups()
        return timestamp, log_level, message
    return None
