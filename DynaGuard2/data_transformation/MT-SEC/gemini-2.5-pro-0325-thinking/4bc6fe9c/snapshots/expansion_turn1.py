from typing import Dict, Any

def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile dictionary.

    Args:
        uid: The unique identifier for the user.
        details: A dictionary containing user attributes.

    Returns:
        A dictionary representing the user profile, combining uid and details.
    """
    profile = {'uid': uid}
    profile.update(details)
    return profile
