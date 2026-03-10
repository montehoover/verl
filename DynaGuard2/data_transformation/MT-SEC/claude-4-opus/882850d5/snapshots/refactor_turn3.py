"""
User activity logging module.

This module provides functionality to record and track user interactions
within a system by maintaining an activity log.
"""

activity_log = []


def create_log_entry(user_alias, interaction_desc):
    """
    Create a log entry dictionary with user and action information.
    
    This function creates a structured log entry containing the user's
    identifier and a description of their action. The entry is formatted
    as a dictionary for easy storage and retrieval.
    
    Args:
        user_alias (str): A unique identifier for the user who performed 
            the action. This could be a username, user ID, or any other
            unique identifier.
        interaction_desc (str): A textual description of the action taken 
            by the user. This should be descriptive enough to understand
            what the user did.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'user': The user's unique identifier
            - 'action': The description of the user's action
    
    Example:
        >>> entry = create_log_entry("john_doe", "Updated profile picture")
        >>> print(entry)
        {'user': 'john_doe', 'action': 'Updated profile picture'}
    """
    return {
        'user': user_alias,
        'action': interaction_desc
    }


def add_entry_to_log(log_entry, log):
    """
    Add a log entry to the provided log list.
    
    This function attempts to append a new log entry to the specified
    log list. It includes error handling to ensure the operation's
    success can be verified.
    
    Args:
        log_entry (dict): The log entry to add. Should be a dictionary
            containing user and action information.
        log (list): The log list to append to. This should be a mutable
            list that will store the log entries.
    
    Returns:
        bool: True if the entry was successfully added to the log,
            False if the operation failed for any reason.
    
    Note:
        The function uses a try-except block to catch any potential
        errors during the append operation, though list.append() rarely
        fails in practice.
    """
    try:
        log.append(log_entry)
        return True
    except Exception:
        return False


def save_user_interaction(user_alias, interaction_desc):
    """
    Record user actions in a system by adding new entries to an activity log.
    
    This is the main function for logging user interactions. It creates
    a structured log entry from the provided information and adds it to
    the global activity log. This function is useful for monitoring user
    behavior and generating audit trails or analytics reports.
    
    Args:
        user_alias (str): A unique identifier for the user who performed 
            the action. This could be a username, user ID, or any other
            unique identifier that distinguishes the user.
        interaction_desc (str): A textual description of the action taken 
            by the user. This should clearly describe what the user did
            in the system.
    
    Returns:
        bool: Returns True if the log entry was successfully added to the
            activity log, False if it was not added due to any error or
            exceeding limits.
    
    Example:
        >>> success = save_user_interaction("alice", "Logged in to system")
        >>> print(success)
        True
        
        >>> save_user_interaction("bob", "Downloaded report.pdf")
        True
    
    Global Variables:
        Uses the global `activity_log` list to store user activity records.
    """
    log_entry = create_log_entry(user_alias, interaction_desc)
    return add_entry_to_log(log_entry, activity_log)
