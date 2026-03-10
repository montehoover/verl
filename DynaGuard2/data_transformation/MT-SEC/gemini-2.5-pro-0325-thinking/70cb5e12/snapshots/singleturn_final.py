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
    # Regex to match the log entry format '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    # Breakdown:
    # \[(.*?)\]  - Captures the TIMESTAMP inside square brackets (non-greedy)
    # \s          - Matches a single whitespace character
    # (.*?)       - Captures the LOG_LEVEL (non-greedy)
    # :           - Matches the colon and space after LOG_LEVEL
    # \s          - Matches a single whitespace character
    # (.*)        - Captures the MESSAGE (greedy, takes the rest of the string)
    log_pattern = r"\[(.*?)\]\s(.*?):\s(.*)"

    match = re.match(log_pattern, record)

    if match:
        # Extract the captured groups
        timestamp = match.group(1)
        log_level = match.group(2)
        message = match.group(3)
        return (timestamp, log_level, message)
    else:
        # Return None if the log entry does not match the format
        return None
