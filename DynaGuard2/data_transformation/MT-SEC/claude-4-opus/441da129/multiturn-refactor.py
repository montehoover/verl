# Global activity log list that stores all user activity records
# This serves as the central repository for tracking user interactions in the system
activity_log = []


def _is_log_full(log, max_size=10000):
    """
    Checks if the activity log has reached its maximum size.
    
    This is a pure function that validates whether the log can accept new entries.
    It helps prevent unbounded growth of the activity log which could lead to
    memory issues in long-running applications.
    
    Args:
        log (list): The activity log to check.
        max_size (int): Maximum allowed size of the log. Defaults to 10000.
    
    Returns:
        bool: True if log is full (has reached or exceeded max_size), 
              False otherwise.
    """
    return len(log) >= max_size


def _create_log_entry(user_key, action_details):
    """
    Creates a log entry dictionary from user key and action details.
    
    This pure function standardizes the format of log entries, ensuring
    consistent structure across all logged activities. This makes it easier
    to process and analyze the logs later.
    
    Args:
        user_key (str): A unique string identifier associated with the user.
        action_details (str): A textual description of the action.
    
    Returns:
        dict: A dictionary containing the log entry data with keys:
              - 'user_key': The user's unique identifier
              - 'action_details': Description of the action performed
    """
    return {
        'user_key': user_key,
        'action_details': action_details
    }


def log_user_event(user_key, action_details):
    """
    Registers user actions in a system by adding entries to an activity log list.
    
    This function is crucial for tracking user interactions, facilitating system 
    audits, and generating usage reports. It ensures that the activity log doesn't
    grow beyond a reasonable size by checking capacity before adding new entries.
    
    The function uses helper functions to maintain clean separation of concerns:
    - _is_log_full: Handles validation logic
    - _create_log_entry: Handles entry creation logic
    
    Args:
        user_key (str): A unique string identifier associated with the user 
                       performing the action.
        action_details (str): A textual description detailing the specific 
                             action undertaken by the user.
    
    Returns:
        bool: Returns True if the activity was successfully logged, 
              False if the logging attempt was rejected due to size limitations.
    
    Example:
        >>> log_user_event("user123", "Viewed dashboard")
        True
        >>> log_user_event("user456", "Updated profile settings")
        True
    """
    # Check if activity log has reached its size limit
    # This prevents memory issues and ensures system stability
    if _is_log_full(activity_log):
        return False
    
    # Create a properly formatted log entry using the helper function
    log_entry = _create_log_entry(user_key, action_details)
    
    # Append the new entry to the global activity log
    activity_log.append(log_entry)
    
    # Return success status
    return True
