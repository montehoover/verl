import re

def find_keyword_in_log(log_line: str, keyword: str) -> bool:
    """
    Checks if a log line contains a particular keyword.

    Args:
        log_line: The log line string to search within.
        keyword: The keyword string to search for.

    Returns:
        True if the keyword is present in the log line, False otherwise.
    """
    return keyword in log_line


def parse_log_line(log_line: str) -> dict:
    """
    Parses a log line into its components.

    Assumes the log line is formatted as '[TIMESTAMP] LEVEL: MESSAGE'.

    Args:
        log_line: The log line string to parse.

    Returns:
        A dictionary with keys 'timestamp', 'level', and 'message'.
        Returns an empty dictionary if parsing fails.
    """
    try:
        # Remove brackets from timestamp
        timestamp_part, rest = log_line.split('] ', 1)
        timestamp = timestamp_part[1:]

        # Split level and message
        level, message = rest.split(': ', 1)

        return {
            "timestamp": timestamp,
            "level": level,
            "message": message.strip()  # Remove leading/trailing whitespace from message
        }
    except ValueError:
        # Handle cases where the log line doesn't match the expected format
        return {
            "timestamp": "",
            "level": "",
            "message": log_line # Or return the original line as message
        }


def extract_log_data(log_line: str):
    """
    Extracts detailed information from log entries using a regular expression.

    Parses log lines formatted as '[TIMESTAMP] LOG_LEVEL: MESSAGE'.

    Args:
        log_line: The log line string to parse.

    Returns:
        A tuple containing (timestamp, log_level, message) if parsing is successful,
        or None if it fails.
    """
    # Regex to capture timestamp, log_level, and message
    # Format: [TIMESTAMP] LOG_LEVEL: MESSAGE
    log_pattern = re.compile(r"^\[(.*?)\]\s(\w+):\s(.*)$")
    match = log_pattern.match(log_line)

    if match:
        timestamp = match.group(1)
        log_level = match.group(2)
        message = match.group(3)
        return (timestamp, log_level, message)
    else:
        return None
