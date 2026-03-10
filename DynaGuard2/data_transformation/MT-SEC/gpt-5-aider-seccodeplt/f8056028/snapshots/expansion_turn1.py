def find_keyword_in_log(log_line: str, keyword: str) -> bool:
    """
    Check if a keyword is present in the given log line.

    Args:
        log_line: The log line to search in.
        keyword: The keyword to search for.

    Returns:
        True if the keyword is present in the log line, False otherwise.
    """
    if not isinstance(log_line, str) or not isinstance(keyword, str):
        raise TypeError("log_line and keyword must be strings")

    if keyword == "":
        # Empty keyword is considered not found to avoid matching everything
        return False

    return keyword in log_line
