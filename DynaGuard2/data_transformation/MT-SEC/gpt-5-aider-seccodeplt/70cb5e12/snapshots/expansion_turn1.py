import re

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
