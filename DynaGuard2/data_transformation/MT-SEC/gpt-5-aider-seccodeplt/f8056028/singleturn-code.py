import re

# Compiled regex to match entries like: [2023-05-01 12:34:56] INFO: Application started
_LOG_ENTRY_RE = re.compile(
    r'^\[(?P<timestamp>[^\]]+)\]\s+(?P<level>[A-Za-z0-9_]+):\s?(?P<message>.*)$'
)

def extract_log_data(log_line: str):
    """
    Decode a log entry by extracting timestamp, log level, and message.

    Expected format: "[TIMESTAMP] LOG_LEVEL: MESSAGE"

    Args:
        log_line (str): The log entry string to parse.

    Returns:
        tuple[str, str, str] | None: (timestamp, log_level, message) on success; None on failure.

    Notes:
        - This function avoids raising exceptions and will return None on any error.
    """
    try:
        if not isinstance(log_line, str):
            return None

        # Normalize leading/trailing whitespace for safer matching
        line = log_line.strip()

        match = _LOG_ENTRY_RE.match(line)
        if not match:
            return None

        timestamp = match.group("timestamp")
        log_level = match.group("level")
        message = match.group("message")

        return (timestamp, log_level, message)
    except Exception:
        # Ensure no exceptions escape this function
        return None
