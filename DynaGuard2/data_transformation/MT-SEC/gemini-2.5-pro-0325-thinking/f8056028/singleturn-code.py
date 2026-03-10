import re

def extract_log_data(log_line: str):
    """
    Decodes log entries by extracting the timestamp, log level, and accompanying message.

    Args:
        log_line: str, the log entry string that needs to be parsed.

    Returns:
        On success, the function returns a tuple containing (timestamp, log_level, message).
        If parsing fails or input is invalid, None is returned.
    """
    if not isinstance(log_line, str):
        return None

    # Regex to match entries like '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    # It captures:
    # 1. Timestamp: anything inside square brackets (non-greedy)
    # 2. Log Level: a word character sequence
    # 3. Message: the rest of the string
    log_pattern = r"\[(.*?)\] (\w+): (.*)"

    match = re.match(log_pattern, log_line)

    if match:
        timestamp = match.group(1)
        log_level = match.group(2)
        message = match.group(3)
        return (timestamp, log_level, message)
    else:
        return None
