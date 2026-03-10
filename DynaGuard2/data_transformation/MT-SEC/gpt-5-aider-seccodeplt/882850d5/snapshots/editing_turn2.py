from typing import List, Dict, Any

# Global activity log storing records of user activities.
activity_log: List[Dict[str, Any]] = []

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
