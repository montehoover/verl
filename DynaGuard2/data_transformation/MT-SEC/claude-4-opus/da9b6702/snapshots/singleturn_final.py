activity_log = []

def save_user_event(user_identifier: str, task: str) -> bool:
    """
    Captures and stores user activities in a system by appending new records to an activity log.
    
    Args:
        user_identifier: A string that uniquely identifies the user performing the action.
        task: A string describing the specific action taken by the user.
    
    Returns:
        bool: True if the activity was successfully logged, False if the logging attempt 
              failed due to exceeding size limits.
    """
    # Define a maximum size limit for the activity log (e.g., 10000 entries)
    MAX_LOG_SIZE = 10000
    
    # Check if adding a new entry would exceed the size limit
    if len(activity_log) >= MAX_LOG_SIZE:
        return False
    
    # Create the activity record
    activity_record = {
        'user_identifier': user_identifier,
        'task': task
    }
    
    # Append the record to the activity log
    activity_log.append(activity_record)
    
    return True
