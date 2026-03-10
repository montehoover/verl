event_log = []

def log_event(event_description: str, user_id: str) -> bool:
    """Appends a user-specific event to the global event log.

    Args:
        event_description: A string describing the event.
        user_id: A string representing the user.

    Returns:
        True if the event was logged successfully, False if the log is full.
    """
    if len(event_log) >= 100:
        return False
    event_log.append((user_id, event_description))
    return True
