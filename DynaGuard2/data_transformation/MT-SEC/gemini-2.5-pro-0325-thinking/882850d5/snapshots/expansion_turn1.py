activity_log = []

def log_entry(entry_message: str) -> None:
    """
    Appends an entry message to the global activity_log.

    Args:
        entry_message: The string message to log.
    """
    global activity_log
    activity_log.append(entry_message)

if __name__ == '__main__':
    # Example usage:
    log_entry("User logged in.")
    log_entry("User viewed page X.")
    log_entry("User logged out.")
    print(f"Activity Log: {activity_log}")
