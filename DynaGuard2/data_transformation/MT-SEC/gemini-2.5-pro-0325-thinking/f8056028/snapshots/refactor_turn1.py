import re

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
        # Regex to capture timestamp, log_level, and message
        # Timestamp: anything within square brackets
        # Log Level: one or more word characters (letters, numbers, underscore)
        # Message: anything after 'LOG_LEVEL: '
        log_pattern = re.compile(r"\[(.*?)\]\s+(\w+):\s+(.*)")
        match = log_pattern.match(log_line)
        if match:
            timestamp, log_level, message = match.groups()
            return timestamp, log_level, message
        else:
            return None
    except Exception:
        # Catch any potential exceptions during regex processing or other issues
        return None
