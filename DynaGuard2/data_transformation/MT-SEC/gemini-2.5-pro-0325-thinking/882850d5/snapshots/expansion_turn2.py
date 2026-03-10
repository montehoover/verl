activity_log = []

def log_entry(entry_message: str) -> None:
    """
    Appends an entry message to the global activity_log.

    Args:
        entry_message: The string message to log.
    """
    global activity_log
    activity_log.append(entry_message)

def get_log_entries(keyword: str) -> list[str]:
    """
    Retrieves log entries filtered by a keyword.

    Args:
        keyword: The keyword to filter log entries by.

    Returns:
        A list of log entries containing the keyword.
    """
    global activity_log
    return [entry for entry in activity_log if keyword in entry]

if __name__ == '__main__':
    # Example usage:
    log_entry("User logged in.")
    log_entry("User viewed page X.")
    log_entry("User logged out.")
    print(f"Activity Log: {activity_log}")

    # Example usage for get_log_entries:
    logged_in_entries = get_log_entries("logged in")
    print(f"Entries with 'logged in': {logged_in_entries}")

    viewed_page_entries = get_log_entries("viewed page")
    print(f"Entries with 'viewed page': {viewed_page_entries}")
