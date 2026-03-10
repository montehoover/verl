from typing import List, Dict, Any, Optional

# Global activity log storing records of user activities.
activity_log: List[Dict[str, Any]] = []

# Optional maximum allowed number of entries in the activity log.
# Set to an integer to enforce a limit, or leave as None for no limit.
ACTIVITY_LOG_MAX_ENTRIES: Optional[int] = None

def count_user_activities(user_alias: str) -> int:
    """
    Return the number of activity entries associated with the given user_alias
    from the global activity_log.

    Each entry in activity_log is expected to be a dict containing:
      - 'user_alias': str
      - 'interaction_desc': str

    Args:
        user_alias: The identifier of the user.

    Returns:
        The count of activity entries for that user.
    """
    return sum(
        1
        for entry in activity_log
        if isinstance(entry, dict)
        and entry.get("user_alias") == user_alias
        and "interaction_desc" in entry
    )

def save_user_interaction(user_alias: str, interaction_desc: str) -> bool:
    """
    Add an activity entry to the global activity_log.

    Args:
        user_alias: A unique user identifier.
        interaction_desc: Description of the user's action.

    Returns:
        True if the entry was successfully added, or False if the log has reached
        its allowed limit or if inputs are invalid.
    """
    if not isinstance(user_alias, str) or not isinstance(interaction_desc, str):
        return False

    if ACTIVITY_LOG_MAX_ENTRIES is not None and len(activity_log) >= ACTIVITY_LOG_MAX_ENTRIES:
        return False

    activity_log.append({
        "user_alias": user_alias,
        "interaction_desc": interaction_desc,
    })
    return True
