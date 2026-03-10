import re


def find_keyword_in_log(log_line, keyword):
    """
    Check if a log line contains a particular keyword.
    
    Args:
        log_line (str): The log line to search in
        keyword (str): The keyword to search for
        
    Returns:
        bool: True if keyword is found in log_line, False otherwise
    """
    return keyword in log_line


def parse_log_line(log_line):
    """
    Parse a log line into structured components.
    
    Args:
        log_line (str): The log line formatted as '[TIMESTAMP] LEVEL: MESSAGE'
        
    Returns:
        dict: Dictionary with keys 'timestamp', 'level', and 'message'
    """
    # Find the closing bracket to separate timestamp
    bracket_end = log_line.find(']')
    if bracket_end == -1:
        return {'timestamp': '', 'level': '', 'message': ''}
    
    # Extract timestamp (without brackets)
    timestamp = log_line[1:bracket_end]
    
    # Find the colon to separate level and message
    remaining = log_line[bracket_end + 1:].strip()
    colon_pos = remaining.find(':')
    if colon_pos == -1:
        return {'timestamp': timestamp, 'level': '', 'message': ''}
    
    # Extract level and message
    level = remaining[:colon_pos].strip()
    message = remaining[colon_pos + 1:].strip()
    
    return {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }


def extract_log_data(log_line):
    """
    Extract log data using regular expression parsing.
    
    Args:
        log_line (str): The log line formatted as '[TIMESTAMP] LOG_LEVEL: MESSAGE'
        
    Returns:
        tuple: (timestamp, log_level, message) if successful, None if parsing fails
    """
    try:
        # Regex pattern to match [TIMESTAMP] LOG_LEVEL: MESSAGE
        pattern = r'^\[([^\]]+)\]\s+([^:]+):\s+(.*)$'
        match = re.match(pattern, log_line)
        
        if match:
            timestamp = match.group(1)
            log_level = match.group(2)
            message = match.group(3)
            return (timestamp, log_level, message)
        else:
            return None
    except:
        return None
