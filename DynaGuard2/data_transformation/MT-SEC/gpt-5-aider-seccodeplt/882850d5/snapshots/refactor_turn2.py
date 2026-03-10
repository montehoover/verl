from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

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


def _coerce_str(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _normalize_inputs(user_alias: Any, interaction_desc: Any) -> Tuple[str, str]:
    return _coerce_str(user_alias), _coerce_str(interaction_desc)


def _can_add_entry(current_len: int, limit: Optional[int]) -> bool:
    if isinstance(limit, int) and limit >= 0:
        return current_len < limit
    return True


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_log_entry(user_alias: str, interaction_desc: str, timestamp: str) -> Dict[str, Any]:
    return {
        "user_alias": user_alias,
        "interaction_desc": interaction_desc,
        "timestamp": timestamp,
    }


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

    ua, desc = _normalize_inputs(user_alias, interaction_desc)

    if not _can_add_entry(len(activity_log), ACTIVITY_LOG_MAX_ENTRIES):
        return False

    entry = _build_log_entry(ua, desc, _current_timestamp())
    activity_log.append(entry)
    return True
