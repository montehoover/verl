import re

def extract_log_data(log_line: str):
    """
    Decode log entries by extracting timestamp, log level, and message.
    
    Args:
        log_line: str - The log entry string that needs to be parsed
        
    Returns:
        On success, returns a tuple containing (timestamp, log_level, message).
        If parsing fails or input is invalid, None is returned.
    """
    try:
        # Regular expression to match the log format: [TIMESTAMP] LOG_LEVEL: MESSAGE
        # Pattern breakdown:
        # \[([^\]]+)\] - Captures content inside square brackets (timestamp)
        # \s+ - Matches one or more whitespace characters
        # ([A-Z]+) - Captures uppercase letters (log level)
        # :\s* - Matches colon followed by optional whitespace
        # (.+) - Captures the rest of the line (message)
        pattern = r'\[([^\]]+)\]\s+([A-Z]+):\s*(.+)'
        
        match = re.match(pattern, log_line)
        
        if match:
            timestamp = match.group(1)
            log_level = match.group(2)
            message = match.group(3)
            return (timestamp, log_level, message)
        else:
            return None
            
    except:
        # Catch any exceptions and return None as specified
        return None
