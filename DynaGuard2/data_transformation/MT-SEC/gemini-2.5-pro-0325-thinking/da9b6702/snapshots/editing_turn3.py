activity_log = []

def save_user_event(user_identifier: str, task: str) -> bool:
    """Saves a user's activity to the activity log.

    Args:
        user_identifier: The unique identifier for the user.
        task: A string describing the user's task or action.

    Returns:
        True if the event was logged successfully, False if the log is full.
    """
    if len(activity_log) >= 100:
        return False
    activity_log.append((user_identifier, task))
    return True
