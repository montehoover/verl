import re

def analyze_log_data(record):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.
    
    Args:
        record: str, the log entry that needs parsing
        
    Returns:
        Returns a tuple (timestamp, log_level, message) if the log entry is properly formatted.
        Otherwise, return None.
    """
    try:
        # Pattern to match [TIMESTAMP] LOG_LEVEL: MESSAGE
        pattern = r'^\[([^\]]+)\]\s+(\w+):\s+(.*)$'
        match = re.match(pattern, record)
        
        if match:
            timestamp = match.group(1)
            log_level = match.group(2)
            message = match.group(3)
            return (timestamp, log_level, message)
        else:
            return None
    except:
        return None
