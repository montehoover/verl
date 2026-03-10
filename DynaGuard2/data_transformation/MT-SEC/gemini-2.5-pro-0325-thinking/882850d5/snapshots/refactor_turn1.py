activity_log = []

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
        due to exceeding limits (currently, always returns True as no limits are defined).
    """
    # In a real-world scenario, you might want to add a timestamp or more structured data.
    # For example:
    # from datetime import datetime
    # log_entry = {
    #     "timestamp": datetime.utcnow().isoformat(),
    #     "user": user_alias,
    #     "action": interaction_desc
    # }
    # activity_log.append(log_entry)

    # For simplicity, as per the initial request, we'll log a simple string or tuple.
    # Let's use a dictionary for better structure.
    log_entry = {"user_alias": user_alias, "interaction_desc": interaction_desc}
    
    # Placeholder for limit checking (e.g., max log size)
    # MAX_LOG_ENTRIES = 1000
    # if len(activity_log) >= MAX_LOG_ENTRIES:
    #     return False  # Log limit exceeded

    activity_log.append(log_entry)
    return True
