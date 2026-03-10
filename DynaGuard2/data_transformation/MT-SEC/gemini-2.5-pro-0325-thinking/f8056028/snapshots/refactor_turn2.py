import re

# Regex to capture timestamp, log_level, and message
# Timestamp: anything within square brackets
# Log Level: one or more word characters (letters, numbers, underscore)
# Message: anything after 'LOG_LEVEL: '
LOG_ENTRY_PATTERN = re.compile(r"\[(.*?)\]\s+(\w+):\s+(.*)")


def _match_log_entry(log_line: str) -> re.Match | None:
    """
    Matches the log line against the predefined log entry pattern.

    Args:
        log_line: The log entry string to match.

    Returns:
        A regex match object if successful, None otherwise.
    """
    try:
        return LOG_ENTRY_PATTERN.match(log_line)
    except Exception:
        # In case of unexpected errors during matching, though rare with pre-compiled patterns.
        return None


def extract_log_data(log_line: str):
    """
    Decodes log entries by extracting the timestamp, log level, and accompanying message.

    Args:
        log_line: The log entry string that needs to be parsed.
                  Expected format: '[TIMESTAMP] LOG_LEVEL: MESSAGE'

    Returns:
        A tuple containing (timestamp, log_level, message) if parsing is successful.
        None if parsing fails or input is invalid.
    """
    try:
        match = _match_log_entry(log_line)
        if match:
            timestamp, log_level, message = match.groups()
            return timestamp, log_level, message
        else:
            return None
    except Exception:
        # Catch any potential exceptions from .groups() or other issues,
        # ensuring the function adheres to its non-raising requirement.
        return None
