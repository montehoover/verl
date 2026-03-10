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
