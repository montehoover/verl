import re

LOG_ENTRY_PATTERN = re.compile(r"\[(.*?)\]\s(.*?):\s(.*)")

def _parse_log_entry(record: str, pattern):
    """
    Matches the log record against the given regex pattern.

    Args:
        record: The log entry string.
        pattern: The compiled regex pattern to use for matching.

    Returns:
        A match object if the record matches the pattern, otherwise None.
    """
    return pattern.match(record)

def _extract_log_components(match_obj):
    """
    Extracts timestamp, log_level, and message from a regex match object.

    Args:
        match_obj: A regex match object.

    Returns:
        A tuple (timestamp, log_level, message).
    """
    return match_obj.groups()

def analyze_log_data(record: str):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.

    Args:
        record: str, the log entry that needs parsing.

    Returns:
        A tuple (timestamp, log_level, message) if the log entry is properly formatted.
        Otherwise, return None.
    """
    match = _parse_log_entry(record, LOG_ENTRY_PATTERN)
    if match:
        return _extract_log_components(match)
    return None
