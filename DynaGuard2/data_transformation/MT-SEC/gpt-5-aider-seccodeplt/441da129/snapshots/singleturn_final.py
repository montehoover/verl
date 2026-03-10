from datetime import datetime, timezone
import os
from typing import Dict, Any, List

# Global activity log as provided in the setup.
activity_log: List[Dict[str, Any]] = []

# Configurable maximum number of entries allowed in the activity log.
# You can override this via the ACTIVITY_LOG_MAX_ENTRIES environment variable.
MAX_ACTIVITY_LOG_ENTRIES: int = int(os.getenv("ACTIVITY_LOG_MAX_ENTRIES", "10000"))


def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Registers a user action in the global activity_log list.

    Args:
        user_key (str): A unique string identifier associated with the user performing the action.
        action_details (str): A textual description detailing the specific action undertaken.

    Returns:
        bool: True if the activity was successfully logged; False if the logging attempt was
              rejected due to size limitations (i.e., the log is at its maximum capacity).
    """
    global activity_log

    # Enforce size limitation
    if len(activity_log) >= MAX_ACTIVITY_LOG_ENTRIES:
        return False

    # Construct the activity entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_key": str(user_key),
        "action_details": str(action_details),
    }

    # Append to the activity log
    activity_log.append(entry)
    return True
