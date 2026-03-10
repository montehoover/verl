def identify_log_level(log_line: str, log_level: str) -> bool:
    """
    Checks if a log line contains a specific log level.

    Args:
        log_line: The log line string to check.
        log_level: The log level string (e.g., 'ERROR', 'DEBUG') to search for.

    Returns:
        True if the log_line contains the log_level, False otherwise.
    """
    return log_level in log_line
