import re

def identify_log_level(log_line: str, log_level: str) -> bool:
    """
    Checks if a log line contains a specific log level.

    Args:
        log_line: The log line string to check.
        log_level: The log level string (e.g., 'ERROR', 'DEBUG') to search for.

    Returns:
        True if the log_line contains the log_level, False otherwise.
    """
    return log_level in log_line


def parse_log_components(log_line: str) -> dict:
    """
    Parses a log line into timestamp, level, and message.

    Args:
        log_line: The log line string to parse.
                  Expected format: "[timestamp] LEVEL: message"

    Returns:
        A dictionary with keys 'timestamp', 'level', and 'message'.
        If parsing fails, values for these keys will be None.
    """
    # Regex to capture:
    # 1. Timestamp: anything within the first square brackets (non-greedy).
    # 2. Level: the first word character sequence (e.g., ERROR, INFO)
    #    after the timestamp and before a colon.
    # 3. Message: everything after the first colon that follows the level.
    log_pattern = re.compile(r"\[(.*?)\]\s*(\w+):(.*)")
    match = log_pattern.match(log_line)

    if match:
        timestamp = match.group(1)
        level = match.group(2)
        message = match.group(3)  # Captures everything after the colon.
        return {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
    else:
        return {
            "timestamp": None,
            "level": None,
            "message": None
        }


def analyze_log_data(log_record: str):
    """
    Parses a log record to extract timestamp, log level, and message.

    Args:
        log_record: A string representing a single log entry,
                    expected in the format '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Returns:
        A tuple (timestamp, log_level, message) if parsing is successful.
        Returns None if the log_record does not match the expected format.
        No exceptions are raised.
    """
    # Regex to capture:
    # 1. TIMESTAMP: Content within the first square brackets.
    # 2. LOG_LEVEL: The word characters (e.g., ERROR, INFO) following the timestamp
    #    and preceding a colon.
    # 3. MESSAGE: All characters after the 'LOG_LEVEL: ' part.
    log_pattern = re.compile(r"\[(.*?)\]\s*(\w+):\s*(.*)")
    match = log_pattern.match(log_record)

    if match:
        timestamp = match.group(1)
        log_level = match.group(2)
        message = match.group(3)
        return (timestamp, log_level, message)
    else:
        return None
