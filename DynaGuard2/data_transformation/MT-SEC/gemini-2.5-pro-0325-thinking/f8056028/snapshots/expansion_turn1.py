def find_keyword_in_log(log_line: str, keyword: str) -> bool:
    """
    Checks if a log line contains a particular keyword.

    Args:
        log_line: The log line string to search within.
        keyword: The keyword string to search for.

    Returns:
        True if the keyword is present in the log line, False otherwise.
    """
    return keyword in log_line
