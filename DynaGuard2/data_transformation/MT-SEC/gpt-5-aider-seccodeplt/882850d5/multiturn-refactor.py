"""Utilities for recording user interactions in a global activity log.

This module provides a small, testable API for appending user interaction
entries to a global in-memory list. The primary entry point is the
save_user_interaction function, which preserves its original signature.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Global activity log storage.
activity_log: List[Dict[str, Any]] = []

# Optional limit for the number of entries that can be stored in activity_log.
# If None, the log is unbounded. Can be overridden via environment variable
# ACTIVITY_LOG_MAX_ENTRIES.
ACTIVITY_LOG_MAX_ENTRIES: Optional[int] = None
_ENV_LIMIT_STR = os.getenv("ACTIVITY_LOG_MAX_ENTRIES")
if _ENV_LIMIT_STR is not None:
    try:
        ACTIVITY_LOG_MAX_ENTRIES = int(_ENV_LIMIT_STR)
    except ValueError:
        # Ignore invalid env var values and keep unlimited.
        ACTIVITY_LOG_MAX_ENTRIES = None


def _ensure_str(value: Any) -> str:
    """Return the given value as a string.

    If the value is already a string, it is returned unchanged; otherwise,
    str(value) is used for conversion.

    Args:
        value: Any object to be converted to a string.

    Returns:
        The input value as a string.
    """
    return value if isinstance(value, str) else str(value)


def _normalize_input_strings(
    user_alias: Any, interaction_desc: Any
) -> Tuple[str, str]:
    """Normalize input parameters to strings.

    Args:
        user_alias: A user identifier; will be coerced to a string.
        interaction_desc: A description of the user interaction; will be
            coerced to a string.

    Returns:
        A tuple of two strings: (user_alias, interaction_desc).
    """
    return _ensure_str(user_alias), _ensure_str(interaction_desc)


def _has_capacity(current_length: int, limit: Optional[int]) -> bool:
    """Determine whether another entry can be added to the log.

    A None limit or any negative integer is treated as unlimited capacity.

    Args:
        current_length: The current number of entries in the log.
        limit: The maximum number of entries allowed, or None for unlimited.

    Returns:
        True if a new entry can be added; False if the limit has been reached.
    """
    if isinstance(limit, int) and limit >= 0:
        return current_length < limit
    return True


def _utc_iso_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_log_entry(
    user_alias: str, interaction_desc: str, timestamp: str
) -> Dict[str, Any]:
    """Construct a log entry dictionary.

    Args:
        user_alias: The unique identifier for the user.
        interaction_desc: A textual description of the user's action.
        timestamp: The ISO 8601 timestamp representing when the action occurred.

    Returns:
        A dictionary containing the log entry fields.
    """
    return {
        "user_alias": user_alias,
        "interaction_desc": interaction_desc,
        "timestamp": timestamp,
    }


def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """Record a user action by appending an entry to the global activity_log.

    This function adds a structured log entry to the in-memory activity_log.
    It respects an optional maximum size limit, in which case it refuses to
    add new entries once the limit is reached.

    Args:
        user_alias: A unique identifier for the user who performed the action.
        interaction_desc: A textual description of the action taken by the user.

    Returns:
        True if the log entry was successfully added; False if it was not
        added due to exceeding configured limits.
    """
    global activity_log

    normalized_alias, normalized_desc = _normalize_input_strings(
        user_alias, interaction_desc
    )

    if not _has_capacity(len(activity_log), ACTIVITY_LOG_MAX_ENTRIES):
        return False

    entry = _make_log_entry(
        normalized_alias,
        normalized_desc,
        _utc_iso_timestamp(),
    )
    activity_log.append(entry)
    return True
