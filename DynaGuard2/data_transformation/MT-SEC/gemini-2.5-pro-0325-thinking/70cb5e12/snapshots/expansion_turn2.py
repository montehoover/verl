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
