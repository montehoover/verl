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
    # Create log entry with user alias and interaction description
    log_entry = {
        'user_alias': user_alias,
        'interaction_desc': interaction_desc
    }
    
    # Add the entry to the activity log
    activity_log.append(log_entry)
    
    # Return True to indicate successful addition
    return True
