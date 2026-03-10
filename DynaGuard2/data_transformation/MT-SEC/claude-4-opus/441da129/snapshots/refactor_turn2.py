activity_log = []

def _is_log_full(log, max_size=10000):
    """
    Checks if the activity log has reached its maximum size.
    
    Args:
        log (list): The activity log to check.
        max_size (int): Maximum allowed size of the log.
    
    Returns:
        bool: True if log is full, False otherwise.
    """
    return len(log) >= max_size

def _create_log_entry(user_key, action_details):
    """
    Creates a log entry dictionary from user key and action details.
    
    Args:
        user_key (str): A unique string identifier associated with the user.
        action_details (str): A textual description of the action.
    
    Returns:
        dict: A dictionary containing the log entry data.
    """
    return {
        'user_key': user_key,
        'action_details': action_details
    }

def log_user_event(user_key, action_details):
    """
    Registers user actions in a system by adding entries to an activity log list.
    
    Args:
        user_key (str): A unique string identifier associated with the user performing the action.
        action_details (str): A textual description detailing the specific action undertaken by the user.
    
    Returns:
        bool: Returns True if the activity was successfully logged, False if the logging attempt 
              was rejected due to size limitations.
    """
    # Check if activity log has reached size limit
    if _is_log_full(activity_log):
        return False
    
    # Create and add the log entry
    log_entry = _create_log_entry(user_key, action_details)
    activity_log.append(log_entry)
    
    return True
