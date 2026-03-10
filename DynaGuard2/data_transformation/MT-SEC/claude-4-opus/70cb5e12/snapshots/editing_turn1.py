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
