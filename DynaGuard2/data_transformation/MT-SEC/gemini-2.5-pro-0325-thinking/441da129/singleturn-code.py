activity_log = []
MAX_LOG_SIZE = 1000

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
    if len(activity_log) >= MAX_LOG_SIZE:
        return False
    
    log_entry = {
        "user_key": user_key,
        "action_details": action_details
    }
    activity_log.append(log_entry)
    return True
