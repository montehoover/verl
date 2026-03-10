import re
from typing import Dict, Optional, Tuple

def identify_log_level(log_line: str, log_level: str) -> bool:
    """
    Return True if the log_line contains the specified log_level as a distinct token,
    otherwise False. Matching is case-insensitive.
    """
    if not isinstance(log_line, str) or not isinstance(log_level, str):
        return False
    if not log_level:
        return False
    pattern = r'\b{}\b'.format(re.escape(log_level))
    return re.search(pattern, log_line, flags=re.IGNORECASE) is not None


def parse_log_components(log_line: str) -> Dict[str, Optional[str]]:
    """
    Parse a log line into components.

    Expected format:
      [timestamp] LEVEL: message

    - timestamp: text within the first pair of square brackets
    - level: the word immediately before the first colon after the timestamp
    - message: everything after that colon (may include additional colons)

    Returns a dict with keys: 'timestamp', 'level', 'message'.
    If parsing fails, values are None.
    """
    result: Dict[str, Optional[str]] = {"timestamp": None, "level": None, "message": None}
    if not isinstance(log_line, str):
        return result

    pattern = r'^\s*\[(?P<timestamp>[^\]]+)\]\s*(?P<level>[A-Za-z]+)\s*:\s*(?P<message>.*)$'
    match = re.match(pattern, log_line)
    if not match:
        return result

    return {
        "timestamp": match.group("timestamp"),
        "level": match.group("level"),
        "message": match.group("message"),
    }


def analyze_log_data(record: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a log record formatted as:
      [TIMESTAMP] LOG_LEVEL: MESSAGE

    Returns a tuple (timestamp, log_level, message) if parsing succeeds, else None.
    Guaranteed not to raise exceptions.
    """
    if not isinstance(record, str):
        return None
    try:
        pattern = r'^\s*\[(?P<timestamp>[^\]]+)\]\s*(?P<log_level>[A-Za-z]+)\s*:\s*(?P<message>.*)$'
        match = re.match(pattern, record)
        if not match:
            return None
        return (
            match.group("timestamp"),
            match.group("log_level"),
            match.group("message"),
        )
    except Exception:
        return None
