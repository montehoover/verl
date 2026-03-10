activity_log = []
MAX_LOG_SIZE = 1000  # Define a maximum size for the log

def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Records user actions in a system by adding new entries to an activity log list.

    This function is useful for monitoring user behavior and generating audit trails
    or analytics reports.

    Args:
        user_alias: A unique identifier for the user who performed the action.
        interaction_desc: A textual description of the action taken by the user.

    Returns:
        True if the log entry was successfully added, False if it was not added
        due to exceeding limits.
    """
    global activity_log
    if len(activity_log) >= MAX_LOG_SIZE:
        return False
    
    log_entry = {
        "user_alias": user_alias,
        "interaction_desc": interaction_desc
    }
    activity_log.append(log_entry)
    return True
