import re


def _parse_log_pattern(log_line):
    """Parse a log line using the standard log format pattern.
    
    Args:
        log_line (str): The log entry string to parse.
        
    Returns:
        re.Match: Match object if successful, None otherwise.
    """
    pattern = r'^\[([^\]]+)\]\s+([A-Z]+):\s+(.*)$'
    return re.match(pattern, log_line)


def extract_log_data(log_line):
    """Decode log entries by extracting the timestamp, log level, and message.
    
    Extracts components from log entries that follow the format:
    '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    
    Args:
        log_line (str): The log entry string that needs to be parsed.
        
    Returns:
        tuple: A tuple containing (timestamp, log_level, message) if parsing
            is successful, None if parsing fails or input is invalid.
            
    Note:
        This function does not raise exceptions and returns None for any
        error condition.
    """
    try:
        match = _parse_log_pattern(log_line)
        
        if not match:
            return None
            
        timestamp = match.group(1)
        log_level = match.group(2)
        message = match.group(3)
        return (timestamp, log_level, message)
            
    except:
        # Return None for any exception
        return None
