activity_log = []
MAX_LOG_SIZE = 1000

def _is_log_capacity_exceeded(current_log: list, max_size: int) -> bool:
    """Checks if the activity log has reached its maximum capacity."""
    return len(current_log) >= max_size

def _create_log_entry(user_key: str, action_details: str) -> dict:
    """Creates a log entry dictionary."""
    return {
        "user_key": user_key,
        "action_details": action_details
    }

def log_user_event(user_key: str, action_details: str) -> bool:
    """Registers user actions in a system by adding entries to an activity log list.

    This function is crucial for tracking user interactions, facilitating system audits,
    and generating usage reports.

    Args:
        user_key: A unique string identifier associated with the user performing the action.
        action_details: A textual description detailing the specific action undertaken by the user.

    Returns:
        True if the activity was successfully logged, False if the logging attempt
        was rejected due to size limitations.
    """
    if _is_log_capacity_exceeded(activity_log, MAX_LOG_SIZE):
        return False
    
    log_entry = _create_log_entry(user_key, action_details)
    activity_log.append(log_entry)
    return True
