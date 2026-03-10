from typing import List, Dict, Any

# Global activity log storing records of user activities.
activity_log: List[Dict[str, Any]] = []

def get_user_activities(user_alias: str) -> List[str]:
    """
    Retrieve a list of interaction descriptions for the specified user_alias
    from the global activity_log.

    Each entry in activity_log is expected to be a dict containing:
      - 'user_alias': str
      - 'interaction_desc': str

    Args:
        user_alias: The identifier of the user.

    Returns:
        A list of interaction descriptions (strings) associated with the user.
    """
    return [
        entry["interaction_desc"]
        for entry in activity_log
        if isinstance(entry, dict)
        and entry.get("user_alias") == user_alias
        and "interaction_desc" in entry
    ]
