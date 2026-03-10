import re


def _get_log_pattern():
    """
    Returns the regex pattern for parsing log entries.
    
    The pattern matches log entries in the format:
    [TIMESTAMP] LOG_LEVEL: MESSAGE
    
    Returns:
        str: Regular expression pattern for log entry matching.
    """
    return r'^\[([^\]]+)\]\s+(\w+):\s+(.*)$'


def _parse_log_entry(record, pattern):
    """
    Attempts to parse a log entry using the given pattern.
    
    This function safely attempts to match the provided record against
    the regex pattern, handling any potential exceptions.
    
    Args:
        record (str): The log entry to parse.
        pattern (str): The regex pattern to use for matching.
        
    Returns:
        re.Match or None: Match object if successful, None otherwise.
    """
    try:
        return re.match(pattern, record)
    except:
        return None


def _extract_log_components(match):
    """
    Extracts timestamp, log level, and message from a regex match.
    
    This function assumes the match object contains three groups:
    1. Timestamp (everything inside square brackets)
    2. Log level (word characters after brackets)
    3. Message (everything after the colon and space)
    
    Args:
        match (re.Match): Regular expression match object containing
                         the captured groups.
        
    Returns:
        tuple: A tuple containing (timestamp, log_level, message).
    """
    timestamp = match.group(1)
    log_level = match.group(2)
    message = match.group(3)
    
    return (timestamp, log_level, message)


def analyze_log_data(record):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.
    
    This function parses log entries that follow the format:
    [TIMESTAMP] LOG_LEVEL: MESSAGE
    
    Example:
        >>> analyze_log_data("[2023-10-20 14:30:00] ERROR: Failed to connect")
        ('2023-10-20 14:30:00', 'ERROR', 'Failed to connect')
    
    Args:
        record (str): The log entry that needs parsing.
        
    Returns:
        tuple or None: Returns a tuple (timestamp, log_level, message) if the 
                      log entry is properly formatted. Otherwise, returns None.
    """
    pattern = _get_log_pattern()
    match = _parse_log_entry(record, pattern)
    
    if match:
        return _extract_log_components(match)
    else:
        return None
