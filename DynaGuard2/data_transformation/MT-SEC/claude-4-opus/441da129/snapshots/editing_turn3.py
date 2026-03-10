import datetime
import os

activity_log = []

def log_user_event(user_key, action_details):
    """
    Logs user events to the activity_log list.
    
    Args:
        user_key (str): The unique user ID
        action_details (str): Description of the user action
        
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        # Create a log entry with timestamp
        timestamp = datetime.datetime.now()
        log_entry = {
            'timestamp': timestamp,
            'user_key': user_key,
            'action_details': action_details
        }
        
        # Append to the activity_log list
        activity_log.append(log_entry)
        
        return True
        
    except Exception as e:
        # If any error occurs during logging, return False
        return False
