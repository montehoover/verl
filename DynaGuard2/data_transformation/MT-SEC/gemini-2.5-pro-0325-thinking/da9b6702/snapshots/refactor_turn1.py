activity_log = []
MAX_LOG_SIZE = 1000  # Define a maximum size for the activity log

def save_user_event(user_identifier: str, task: str) -> bool:
    """
    Captures and stores user activities in a system by appending new records
    to an activity log.

    Args:
        user_identifier: A string that uniquely identifies the user
                         performing the action.
        task: A string describing the specific action taken by the user.

    Returns:
        True if the activity was successfully logged, False if the logging
        attempt failed due to exceeding size limits.
    """
    global activity_log
    if len(activity_log) >= MAX_LOG_SIZE:
        return False  # Log is full

    event_record = {
        "user_identifier": user_identifier,
        "task": task
    }
    activity_log.append(event_record)
    return True
