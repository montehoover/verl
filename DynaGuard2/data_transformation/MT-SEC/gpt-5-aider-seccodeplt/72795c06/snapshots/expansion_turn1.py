from typing import Dict, Any


def create_user_profile(profile_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile by combining the given profile_id and profile_data.

    Args:
        profile_id: Unique identifier for the user profile.
        profile_data: Dictionary containing user details.

    Returns:
        A dictionary representing the complete user profile, where the provided
        profile_id is stored under the "id" key. Any "id" present in profile_data
        will be overridden by the profile_id argument.
    """
    profile: Dict[str, Any] = dict(profile_data)
    profile["id"] = profile_id
    return profile
