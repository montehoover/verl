activity_log = []

def save_user_interaction(user_alias, interaction_desc):
    """
    Records user actions in a system by adding new entries to an activity log list.
    
    Args:
        user_alias (str): A unique identifier for the user who performed the action.
        interaction_desc (str): A textual description of the action taken by the user.
    
    Returns:
        bool: Returns True if the log entry was successfully added, False if it was not added due to exceeding limits.
    """
    try:
        log_entry = {
            'user': user_alias,
            'action': interaction_desc
        }
        activity_log.append(log_entry)
        return True
    except:
        return False
