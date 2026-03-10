from typing import Dict, List, Optional

# Global activity log repository
activity_log: List[Dict[str, str]] = []


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

    # Optional maximum size limit. If not provided or invalid, no size limit is enforced.
    max_entries: Optional[int] = globals().get("MAX_ACTIVITY_LOG_ENTRIES")
    if isinstance(max_entries, int) and max_entries >= 0:
        if len(activity_log) >= max_entries:
            return False

    # Coerce to strings to satisfy type expectations
    record = {
        "user_key": str(user_key),
        "action_details": str(action_details),
    }

    activity_log.append(record)
    return True
