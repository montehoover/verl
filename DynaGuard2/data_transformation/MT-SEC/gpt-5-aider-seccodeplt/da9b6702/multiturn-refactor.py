"""
Activity logging utilities.

This module provides a small, testable API for recording user activities
into a global in-memory activity log. The main entry point is
`save_user_event`, which validates constraints and appends a structured
record. Helper functions are kept pure to simplify unit testing and
improve maintainability.
"""

from datetime import datetime
import logging
from typing import List, Dict, Any

# Module-level logger for observability.
logger = logging.getLogger(__name__)

# Global activity log repository
activity_log: List[Dict[str, Any]] = []

# Size limits
MAX_LOG_ENTRIES = 10000  # Maximum number of records allowed in the activity log
MAX_TASK_LENGTH = 2000   # Maximum allowed length for the task description (characters)


def within_log_size_limit(log: List[Dict[str, Any]],
                          max_entries: int = MAX_LOG_ENTRIES) -> bool:
    """
    Determine whether another entry can be added to the activity log.

    This is a pure function and does not mutate its inputs.

    Args:
        log (List[Dict[str, Any]]): The current activity log.
        max_entries (int): The maximum allowed number of entries.

    Returns:
        bool: True if another entry can be added, False otherwise.
    """
    return len(log) < max_entries


def append_record_pure(log: List[Dict[str, Any]],
                       record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a new log with the provided record appended.

    This is a pure function: it does not mutate the input log and instead
    returns a new list.

    Args:
        log (List[Dict[str, Any]]): The current activity log.
        record (Dict[str, Any]): The activity record to append.

    Returns:
        List[Dict[str, Any]]: A new list containing the existing log records
        plus the new record.
    """
    return log + [record]


def save_user_event(user_identifier: str, task: str) -> bool:
    """
    Capture and store a user activity by appending a new record to the
    global activity log.

    The function enforces simple constraints to prevent unbounded growth
    and excessively large task descriptions. It logs when it is called,
    and whether the attempt was successful.

    Args:
        user_identifier (str): Uniquely identifies the user performing the action.
        task (str): Description of the specific action taken by the user.

    Returns:
        bool: True if the activity was successfully logged; False if the
        logging attempt failed due to exceeding size limits (either the
        log is at capacity or the task length exceeds the configured limit).

    Side Effects:
        Mutates the global `activity_log` by appending a new record on success.
    """
    global activity_log

    logger.info(
        "save_user_event called for user_identifier=%r", user_identifier
    )

    # Enforce size limits
    if not within_log_size_limit(activity_log, MAX_LOG_ENTRIES):
        logger.warning(
            "save_user_event rejected: activity log at capacity (%d/%d).",
            len(activity_log),
            MAX_LOG_ENTRIES,
        )
        return False

    if isinstance(task, str) and len(task) > MAX_TASK_LENGTH:
        logger.warning(
            "save_user_event rejected: task length %d exceeds limit %d.",
            len(task),
            MAX_TASK_LENGTH,
        )
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

    # Use pure append to get a new log and rebind the global reference
    activity_log = append_record_pure(activity_log, record)

    logger.info(
        "save_user_event success: appended record for user_identifier=%r; new size=%d",
        user_identifier,
        len(activity_log),
    )
    return True
