import re

def identify_log_level(log_line, log_level):
    """
    Check if a log line contains the specified log level.
    
    Args:
        log_line (str): The log line to check
        log_level (str): The log level to search for (e.g., 'ERROR', 'DEBUG')
    
    Returns:
        bool: True if the log line contains the log level, False otherwise
    """
    return log_level in log_line

def parse_log_components(log_line):
    """
    Parse a log line into structured components.
    
    Args:
        log_line (str): The log line to parse
        
    Returns:
        dict: A dictionary with keys 'timestamp', 'level', and 'message'
    """
    # Extract timestamp within square brackets
    timestamp_match = re.search(r'\[(.*?)\]', log_line)
    timestamp = timestamp_match.group(1) if timestamp_match else None
    
    # Extract level and message separated by colon
    # Find the part after the timestamp
    after_timestamp = log_line
    if timestamp_match:
        after_timestamp = log_line[timestamp_match.end():]
    
    # Split by first colon to get level and message
    parts = after_timestamp.strip().split(':', 1)
    
    if len(parts) >= 2:
        level = parts[0].strip()
        message = parts[1].strip()
    else:
        level = None
        message = after_timestamp.strip()
    
    return {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
