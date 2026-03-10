"""
Activity logging module.

This module provides a simple, maintainable interface for recording user
activity into a global in-memory log. The design favors readability and
testability by separating concerns into small, pure helper functions:

- validate_capacity: Confirms whether the log can accept another record.
- create_activity_record: Builds a normalized record from inputs.
- append_record_immutably: Returns a new log with the record appended.
- log_user_event: Orchestrates validation and logging while preserving
  the required external API.

An optional global configuration key, MAX_ACTIVITY_LOG_ENTRIES, may be
defined by the host environment to limit the number of entries stored in
the log. If not defined, or misconfigured (non-int or negative), the log
is considered to have unlimited capacity.
"""

from typing import Dict, List, Optional, Final

# Type alias for a single activity record.
ActivityRecord = Dict[str, str]

# Name of the optional global that, if present, constrains log capacity.
MAX_ENTRIES_GLOBAL_NAME: Final[str] = "MAX_ACTIVITY_LOG_ENTRIES"

# Global activity log repository. This is the central store of all user
# activity records created by log_user_event.
activity_log: List[ActivityRecord] = []


def validate_capacity(current_size: int, max_entries: Optional[int]) -> bool:
    """
    Determine whether another record can be added to the log based on capacity.

    This function is pure: it relies only on its parameters and has no
    side effects.

    Args:
        current_size (int): The current number of entries in the log.
        max_entries (Optional[int]): The maximum allowed entries in the log.
            If None, non-integer, or a negative value, the capacity is
            considered unlimited.

    Returns:
        bool: True if capacity allows adding another record; False otherwise.
    """
    if isinstance(max_entries, int) and max_entries >= 0:
        return current_size < max_entries
    return True


def create_activity_record(user_key: str, action_details: str) -> ActivityRecord:
    """
    Build a normalized activity record from the provided inputs.

    This function is pure and performs minimal normalization by coercing
    inputs to strings.

    Args:
        user_key (str): Unique user identifier associated with the action.
        action_details (str): Description of the specific action performed.

    Returns:
        ActivityRecord: A dictionary representing the activity record.
    """
    return {
        "user_key": str(user_key),
        "action_details": str(action_details),
    }


def append_record_immutably(
    log: List[ActivityRecord],
    record: ActivityRecord,
) -> List[ActivityRecord]:
    """
    Return a new log list with the record appended.

    This function is pure and does not mutate the input list; instead, it
    returns a new list instance that includes the appended record. This
    functional approach can aid testing and reasoning about state changes.

    Args:
        log (List[ActivityRecord]): The existing activity log.
        record (ActivityRecord): The record to append.

    Returns:
        List[ActivityRecord]: A new list containing all prior records plus
        the new record.
    """
    return log + [record]


def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Register a user action in the global activity log.

    This function coordinates capacity validation and record creation, then
    appends the record to the global activity_log. The function signature is
    intentionally kept simple for external callers.

    Capacity control:
        If a global named MAX_ACTIVITY_LOG_ENTRIES exists and is a
        non-negative integer, it constrains the maximum number of records
        that can be logged. When the capacity is reached, the function
        returns False without logging the event. If the global is missing,
        not an int, or negative, capacity is treated as unlimited.

    Args:
        user_key (str): A unique string identifier for the user performing
            the action.
        action_details (str): A textual description of the user's action.

    Returns:
        bool: True if the activity was successfully logged; False if the
        logging attempt was rejected due to size limitations.
    """
    global activity_log

    # Resolve optional maximum size limit from global configuration.
    max_entries: Optional[int] = globals().get(MAX_ENTRIES_GLOBAL_NAME)

    # Validate capacity using a pure helper.
    if not validate_capacity(len(activity_log), max_entries):
        return False

    # Build the record using a pure helper.
    record = create_activity_record(user_key, action_details)

    # Append immutably and update the global log.
    activity_log = append_record_immutably(activity_log, record)
    return True
