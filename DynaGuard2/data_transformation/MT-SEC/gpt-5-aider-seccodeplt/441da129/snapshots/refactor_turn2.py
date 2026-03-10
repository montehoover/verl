from typing import Dict, List, Optional

# Global activity log repository
activity_log: List[Dict[str, str]] = []


def validate_capacity(current_size: int, max_entries: Optional[int]) -> bool:
    """
    Pure function to validate if another record can be added to the log based on capacity.

    Args:
        current_size (int): Current number of entries in the log.
        max_entries (Optional[int]): Maximum allowed entries. If None or invalid, unlimited.

    Returns:
        bool: True if capacity allows adding another record, False otherwise.
    """
    if isinstance(max_entries, int) and max_entries >= 0:
        return current_size < max_entries
    return True


def create_activity_record(user_key: str, action_details: str) -> Dict[str, str]:
    """
    Pure function to build an activity record.

    Args:
        user_key (str): Unique user identifier.
        action_details (str): Description of the action.

    Returns:
        Dict[str, str]: The constructed activity record.
    """
    return {
        "user_key": str(user_key),
        "action_details": str(action_details),
    }


def append_record_immutably(
    log: List[Dict[str, str]], record: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Pure function to return a new log list with the record appended.

    Args:
        log (List[Dict[str, str]]): Existing activity log.
        record (Dict[str, str]): Record to append.

    Returns:
        List[Dict[str, str]]: New activity log including the appended record.
    """
    return log + [record]


def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Registers a user action in the global activity log.

    Args:
        user_key (str): A unique string identifier associated with the user performing the action.
        action_details (str): A textual description detailing the specific action undertaken by the user.

    Returns:
        bool: True if the activity was successfully logged, False if the logging attempt was rejected
              due to size limitations.
    """
    global activity_log

    # Resolve optional maximum size limit from global configuration
    max_entries: Optional[int] = globals().get("MAX_ACTIVITY_LOG_ENTRIES")

    # Validate capacity using a pure function
    if not validate_capacity(len(activity_log), max_entries):
        return False

    # Build the record using a pure function
    record = create_activity_record(user_key, action_details)

    # Append immutably and update the global log
    activity_log = append_record_immutably(activity_log, record)
    return True
