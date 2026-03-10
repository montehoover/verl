from datetime import datetime
from typing import Optional

# Global activity log storage
activity_log = []

# Optional maximum number of entries allowed in the activity log.
# Set to an integer (e.g., 10000) to enforce a limit, or None for no limit.
MAX_ACTIVITY_LOG_ENTRIES: Optional[int] = None


def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Records a user action by adding a new entry to the global activity_log list.

    Args:
        user_alias: A unique identifier for the user who performed the action.
        interaction_desc: A textual description of the action taken by the user.

    Returns:
        True if the log entry was successfully added.
        False if it was not added due to exceeding limits (MAX_ACTIVITY_LOG_ENTRIES).
    """
    global activity_log

    # Enforce optional size limit
    if MAX_ACTIVITY_LOG_ENTRIES is not None and len(activity_log) >= MAX_ACTIVITY_LOG_ENTRIES:
        return False

    entry = {
        "user_alias": user_alias,
        "interaction_desc": interaction_desc,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    activity_log.append(entry)
    return True
