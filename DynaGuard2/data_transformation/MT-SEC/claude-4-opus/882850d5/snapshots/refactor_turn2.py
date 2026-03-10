activity_log = []

def create_log_entry(user_alias, interaction_desc):
    """
    Creates a log entry dictionary with user and action information.
    
    Args:
        user_alias (str): A unique identifier for the user who performed the action.
        interaction_desc (str): A textual description of the action taken by the user.
    
    Returns:
        dict: A dictionary containing the user and action information.
    """
    return {
        'user': user_alias,
        'action': interaction_desc
    }

def add_entry_to_log(log_entry, log):
    """
    Adds a log entry to the provided log list.
    
    Args:
        log_entry (dict): The log entry to add.
        log (list): The log list to append to.
    
    Returns:
        bool: True if successfully added, False otherwise.
    """
    try:
        log.append(log_entry)
        return True
    except:
        return False

def save_user_interaction(user_alias, interaction_desc):
    """
    Records user actions in a system by adding new entries to an activity log list.
    
    Args:
        user_alias (str): A unique identifier for the user who performed the action.
        interaction_desc (str): A textual description of the action taken by the user.
    
    Returns:
        bool: Returns True if the log entry was successfully added, False if it was not added due to exceeding limits.
    """
    log_entry = create_log_entry(user_alias, interaction_desc)
    return add_entry_to_log(log_entry, activity_log)
