activity_log = []

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
    # Check if activity log has reached a size limit (e.g., 10000 entries)
    if len(activity_log) >= 10000:
        return False
    
    # Create the log entry
    log_entry = {
        'user_key': user_key,
        'action_details': action_details
    }
    
    # Add the entry to the activity log
    activity_log.append(log_entry)
    
    return True
