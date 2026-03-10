from typing import Dict, Any

def create_user_profile(profile_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining a profile ID and profile data.

    Args:
        profile_id: The unique identifier for the profile.
        profile_data: A dictionary containing the user's details.

    Returns:
        A dictionary representing the user profile, including the profile_id.
    """
    created_profile = profile_data.copy()
    created_profile['profile_id'] = profile_id
    return created_profile
