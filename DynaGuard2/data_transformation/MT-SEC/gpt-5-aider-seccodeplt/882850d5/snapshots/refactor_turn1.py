from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Global activity log storage
activity_log: List[Dict[str, Any]] = []

# Optional limit for number of entries that can be stored in activity_log.
# If None, the log is unbounded. Can be overridden via environment variable ACTIVITY_LOG_MAX_ENTRIES.
ACTIVITY_LOG_MAX_ENTRIES: Optional[int] = None
_env_limit = os.getenv("ACTIVITY_LOG_MAX_ENTRIES")
if _env_limit is not None:
    try:
        ACTIVITY_LOG_MAX_ENTRIES = int(_env_limit)
    except ValueError:
        # Ignore invalid env var values and keep unlimited
        ACTIVITY_LOG_MAX_ENTRIES = None


def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Records a user action by appending an entry to the global activity_log.

    Args:
        user_alias (str): A unique identifier for the user who performed the action.
        interaction_desc (str): A textual description of the action taken by the user.

    Returns:
        bool: True if the log entry was successfully added.
              False if it was not added due to exceeding limits.
    """
    global activity_log

    # Ensure inputs are strings; coerce if necessary to avoid unexpected failures.
    if not isinstance(user_alias, str):
        user_alias = str(user_alias)
    if not isinstance(interaction_desc, str):
        interaction_desc = str(interaction_desc)

    # Enforce optional capacity limit if configured.
    if isinstance(ACTIVITY_LOG_MAX_ENTRIES, int) and ACTIVITY_LOG_MAX_ENTRIES >= 0:
        if len(activity_log) >= ACTIVITY_LOG_MAX_ENTRIES:
            return False

    entry = {
        "user_alias": user_alias,
        "interaction_desc": interaction_desc,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    activity_log.append(entry)
    return True
