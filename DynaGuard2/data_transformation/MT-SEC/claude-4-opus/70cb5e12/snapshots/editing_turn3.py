import re


def extract_timestamp(log_entry):
    """
    Extract timestamp from a log entry in format '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    
    Args:
        log_entry (str): The log entry string
        
    Returns:
        str: The timestamp part of the log entry
    """
    # Find the closing bracket
    end_bracket = log_entry.find(']')
    if end_bracket == -1:
        return ""
    
    # Extract everything between '[' and ']'
    return log_entry[1:end_bracket]


def extract_log_details(log_entry):
    """
    Extract all components from a log entry in format '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    
    Args:
        log_entry (str): The log entry string
        
    Returns:
        dict: Dictionary with keys 'timestamp', 'log_level', and 'message'
    """
    result = {
        'timestamp': '',
        'log_level': '',
        'message': ''
    }
    
    # Find the closing bracket for timestamp
    end_bracket = log_entry.find(']')
    if end_bracket == -1:
        return result
    
    # Extract timestamp
    result['timestamp'] = log_entry[1:end_bracket]
    
    # Find the colon that separates log level from message
    colon_pos = log_entry.find(':', end_bracket)
    if colon_pos == -1:
        return result
    
    # Extract log level (skip the space after ']')
    result['log_level'] = log_entry[end_bracket + 2:colon_pos]
    
    # Extract message (skip the space after ':')
    result['message'] = log_entry[colon_pos + 2:]
    
    return result


def analyze_log_data(record):
    """
    Parse a log entry using regex matching the format '[TIMESTAMP] LOG_LEVEL: MESSAGE'
    
    Args:
        record (str): The log entry string
        
    Returns:
        tuple: (timestamp, log_level, message) if valid, None if invalid
    """
    pattern = r'^\[([^\]]+)\]\s+([^:]+):\s+(.+)$'
    match = re.match(pattern, record)
    
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        return None
