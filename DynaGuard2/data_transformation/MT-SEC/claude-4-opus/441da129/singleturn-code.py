activity_log = []

def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Registers user actions in a system by adding entries to an activity log list.
    
    Args:
        user_key: A unique string identifier associated with the user performing the action.
        action_details: A textual description detailing the specific action undertaken by the user.
    
    Returns:
        bool: Returns True if the activity was successfully logged, False if the logging 
              attempt was rejected due to size limitations.
    """
    # Define a maximum log size to prevent unlimited growth
    MAX_LOG_SIZE = 10000
    
    # Check if adding a new entry would exceed the size limit
    if len(activity_log) >= MAX_LOG_SIZE:
        return False
    
    # Create a log entry (could be a dict, tuple, or formatted string)
    log_entry = {
        'user_key': user_key,
        'action_details': action_details
    }
    
    # Add the entry to the activity log
    activity_log.append(log_entry)
    
    return True
