activity_log = []
MAX_LOG_ENTRIES = 1000  # Example limit, can be adjusted

def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Adds a user interaction entry to the global activity log.

    Args:
        user_alias: The unique identifier for the user.
        interaction_desc: A description of the user's action.

    Returns:
        True if the entry was successfully added, False otherwise (e.g., if log limit is exceeded).
    """
    if len(activity_log) >= MAX_LOG_ENTRIES:
        return False  # Log limit exceeded

    entry = {
        'user_alias': user_alias,
        'interaction_desc': interaction_desc
    }
    activity_log.append(entry)
    return True
