activity_log = []
MAX_LOG_SIZE = 100  # Example maximum log size

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

def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Records a user interaction in the activity log.

    Args:
        user_alias: The alias of the user.
        interaction_desc: The description of the interaction.

    Returns:
        True if the entry was successfully added, False otherwise (e.g., if log is full).
    """
    global activity_log
    if len(activity_log) >= MAX_LOG_SIZE:
        return False  # Log is full
    
    entry_message = f"User {user_alias}: {interaction_desc}"
    activity_log.append(entry_message)
    return True

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

    # Example usage for save_user_interaction:
    interaction1_saved = save_user_interaction("Alice", "clicked button A")
    print(f"Interaction 1 saved: {interaction1_saved}")
    interaction2_saved = save_user_interaction("Bob", "submitted form B")
    print(f"Interaction 2 saved: {interaction2_saved}")
    
    print(f"Updated Activity Log: {activity_log}")

    # Example of log being full (if MAX_LOG_SIZE is small enough for testing)
    # for i in range(MAX_LOG_SIZE + 5):
    #     saved = save_user_interaction(f"User{i}", f"Action {i}")
    #     if not saved:
    #         print(f"Log full. Could not save interaction for User{i}")
    #         break
    # print(f"Final Activity Log count: {len(activity_log)}")
