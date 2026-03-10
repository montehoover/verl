from datetime import datetime

# Global activity log repository
activity_log = []

# Size limits
MAX_LOG_ENTRIES = 10000  # Maximum number of records allowed in the activity log
MAX_TASK_LENGTH = 2000   # Maximum allowed length for the task description (characters)


def save_user_event(user_identifier: str, task: str) -> bool:
    """
    Captures and stores user activities in a system by appending new records to an activity log.

    Args:
        user_identifier (str): A string that uniquely identifies the user performing the action.
        task (str): A string describing the specific action taken by the user.

    Returns:
        bool: True if the activity was successfully logged,
              False if the logging attempt failed due to exceeding size limits.
    """
    global activity_log

    # Enforce size limits
    if len(activity_log) >= MAX_LOG_ENTRIES:
        return False
    if isinstance(task, str) and len(task) > MAX_TASK_LENGTH:
        return False

    # Defensive conversion to strings if different types are passed
    if not isinstance(user_identifier, str):
        user_identifier = str(user_identifier)
    if not isinstance(task, str):
        task = str(task)

    record = {
        "user_identifier": user_identifier,
        "task": task,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    activity_log.append(record)
    return True
