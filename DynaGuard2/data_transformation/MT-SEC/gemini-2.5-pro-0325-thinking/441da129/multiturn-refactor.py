"""
Manages user activity logging with size limitations.

This module provides functionality to log user events, ensuring that the log
does not exceed a predefined capacity. It includes helper functions for
checking log capacity and creating log entries.
"""

activity_log = []  # Global list to store user activity records.
MAX_LOG_SIZE = 1000  # Maximum number of entries allowed in the activity log.


def _is_log_capacity_exceeded(current_log: list, max_size: int) -> bool:
    """
    Checks if the activity log has reached its maximum capacity.

    Args:
        current_log: The current list of log entries.
        max_size: The maximum allowed size for the log.

    Returns:
        True if the log capacity is exceeded, False otherwise.
    """
    return len(current_log) >= max_size


def _create_log_entry(user_key: str, action_details: str) -> dict:
    """
    Creates a structured log entry.

    Args:
        user_key: The unique identifier for the user.
        action_details: A description of the action performed by the user.

    Returns:
        A dictionary representing the log entry.
    """
    return {
        "user_key": user_key,
        "action_details": action_details
    }


def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Registers user actions in a system by adding entries to an activity log list.

    This function is crucial for tracking user interactions, facilitating system audits,
    and generating usage reports.

    Args:
        user_key: A unique string identifier associated with the user performing the action.
        action_details: A textual description detailing the specific action undertaken by the user.

    Returns:
        True if the activity was successfully logged, False if the logging attempt
        was rejected due to size limitations.
    """
    # Check if the log capacity has been reached before adding a new entry.
    if _is_log_capacity_exceeded(activity_log, MAX_LOG_SIZE):
        return False  # Log is full, reject new entry.
    
    # Create a new log entry.
    log_entry = _create_log_entry(user_key, action_details)
    
    # Add the new entry to the activity log.
    activity_log.append(log_entry)
    
    return True  # Log entry successfully added.
