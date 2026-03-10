import re

def _get_log_pattern():
    """Returns the regex pattern for parsing log entries."""
    return r'^\[([^\]]+)\]\s+(\w+):\s+(.*)$'

def _parse_log_entry(record, pattern):
    """
    Attempts to parse a log entry using the given pattern.
    
    Args:
        record: str, the log entry to parse
        pattern: str, the regex pattern to use
        
    Returns:
        re.Match object if successful, None otherwise
    """
    try:
        return re.match(pattern, record)
    except:
        return None

def _extract_log_components(match):
    """
    Extracts timestamp, log level, and message from a regex match.
    
    Args:
        match: re.Match object
        
    Returns:
        tuple of (timestamp, log_level, message)
    """
    timestamp = match.group(1)
    log_level = match.group(2)
    message = match.group(3)
    return (timestamp, log_level, message)

def analyze_log_data(record):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.
    
    Args:
        record: str, the log entry that needs parsing
        
    Returns:
        Returns a tuple (timestamp, log_level, message) if the log entry is properly formatted.
        Otherwise, return None.
    """
    pattern = _get_log_pattern()
    match = _parse_log_entry(record, pattern)
    
    if match:
        return _extract_log_components(match)
    else:
        return None
