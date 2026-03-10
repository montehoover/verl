import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

activity_log = []
MAX_LOG_SIZE = 1000  # Define a maximum size for the activity log


def _is_log_full(current_log: list, max_size: int) -> bool:
    """Checks if the log has reached its maximum size."""
    return len(current_log) >= max_size


def _create_event_record(user_identifier: str, task: str) -> dict:
    """Creates an event record dictionary."""
    return {"user_identifier": user_identifier, "task": task}


def _add_record_to_log(current_log: list, record: dict) -> list:
    """
    Creates a new list with the record appended.
    This ensures purity by not modifying the original list directly.
    """
    return current_log + [record]


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
    logger.info(
        f"Attempting to save event for user '{user_identifier}' with task: '{task}'"
    )

    if _is_log_full(activity_log, MAX_LOG_SIZE):
        logger.warning(
            f"Activity log is full (size: {len(activity_log)}, max: {MAX_LOG_SIZE}). "
            f"Event for user '{user_identifier}' with task '{task}' not saved."
        )
        return False  # Log is full

    event_record = _create_event_record(user_identifier, task)
    activity_log = _add_record_to_log(activity_log, event_record)  # Assign back to the global
    logger.info(
        f"Successfully saved event for user '{user_identifier}' with task: '{task}'. "
        f"Current log size: {len(activity_log)}"
    )
    return True
