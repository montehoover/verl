from typing import Any, Dict, Optional

# Global data structure to keep track of user profiles.
# Populate this dictionary elsewhere in the application.
# Example structure:
# USER_PROFILES = {
#     1: {"id": 1, "name": "Alice"},
#     2: {"id": 2, "name": "Bob"},
# }
USER_PROFILES: Dict[int, Dict[str, Any]] = {}


def can_view_own_profile(user_id: int) -> Any:
    """
    Return the user's profile if they can view it, or 'Unauthorized' otherwise.

    Rules:
    - If the user_id exists in USER_PROFILES, return the profile.
    - Otherwise, return 'Unauthorized'.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict | str: The user's profile dict, or 'Unauthorized'.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    profile: Optional[Dict[str, Any]] = USER_PROFILES.get(user_id)
    if profile is None:
        return "Unauthorized"

    return profile
