import re

def analyze_log_data(record: str):
    """
    Decodes log entries by extracting the timestamp, log level, and message part.
    
    Args:
        record: str, the log entry that needs parsing
        
    Returns:
        Returns a tuple (timestamp, log_level, message) if the log entry is properly formatted.
        Otherwise, return None.
    """
    try:
        # Define the regex pattern to match '[TIMESTAMP] LOG_LEVEL: MESSAGE'
        pattern = r'^\[(.+?)\]\s+(\w+):\s+(.*)$'
        
        # Try to match the pattern
        match = re.match(pattern, record)
        
        if match:
            # Extract the three groups: timestamp, log_level, message
            timestamp = match.group(1)
            log_level = match.group(2)
            message = match.group(3)
            
            return (timestamp, log_level, message)
        else:
            # If no match, return None
            return None
            
    except Exception:
        # Catch any exceptions and return None
        return None
