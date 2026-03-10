event_log = []

def log_event(event_description: str) -> bool:
    """Appends an event to the global event log.

    Args:
        event_description: A string describing the event.

    Returns:
        True if the event was logged successfully.
    """
    event_log.append(event_description)
    return True
