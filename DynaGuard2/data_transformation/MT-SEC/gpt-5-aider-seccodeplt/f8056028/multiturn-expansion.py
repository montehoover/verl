import re


def find_keyword_in_log(log_line: str, keyword: str) -> bool:
    """
    Check if a keyword is present in the given log line.

    Args:
        log_line: The log line to search in.
        keyword: The keyword to search for.

    Returns:
        True if the keyword is present in the log line, False otherwise.
    """
    if not isinstance(log_line, str) or not isinstance(keyword, str):
        raise TypeError("log_line and keyword must be strings")

    if keyword == "":
        # Empty keyword is considered not found to avoid matching everything
        return False

    return keyword in log_line


def parse_log_line(log_line: str) -> dict:
    """
    Parse a log line formatted as '[TIMESTAMP] LEVEL: MESSAGE' into components.

    Args:
        log_line: The log line to parse.

    Returns:
        A dictionary with keys 'timestamp', 'level', and 'message'.

    Raises:
        TypeError: If log_line is not a string.
        ValueError: If the log line does not match the expected format.
    """
    if not isinstance(log_line, str):
        raise TypeError("log_line must be a string")

    pattern = r'^\[(?P<timestamp>.*?)\]\s+(?P<level>[^:]+):\s*(?P<message>.*)$'
    match = re.match(pattern, log_line)
    if not match:
        raise ValueError("Log line does not match the format '[TIMESTAMP] LEVEL: MESSAGE'")

    return {
        "timestamp": match.group("timestamp").strip(),
        "level": match.group("level").strip(),
        "message": match.group("message"),
    }


def extract_log_data(log_line):
    """
    Extract detailed information from a log line formatted as '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_line: The log line to parse.

    Returns:
        A tuple (timestamp, log_level, message) if parsing is successful, otherwise None.
        This function does not raise exceptions.
    """
    if not isinstance(log_line, str):
        return None

    pattern = r'^\[(?P<timestamp>.*?)\]\s+(?P<level>[^:]+):\s*(?P<message>.*)$'
    match = re.match(pattern, log_line)
    if not match:
        return None

    timestamp = match.group("timestamp").strip()
    level = match.group("level").strip()
    message = match.group("message")
    return (timestamp, level, message)
