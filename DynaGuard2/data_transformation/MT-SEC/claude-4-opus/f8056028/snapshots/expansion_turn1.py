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
