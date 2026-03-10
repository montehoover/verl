from typing import Dict, Any, List

existing_profile_ids: List[str] = []

def assign_profile_id(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique profile_id to a profile dictionary and records it.

    Args:
        profile_data: The profile dictionary to update.

    Returns:
        The updated profile dictionary with 'profile_id'.
    """
    # Generate a simple unique ID for demonstration purposes
    profile_id = f"user_{len(existing_profile_ids) + 1}"
    
    # Ensure the generated ID is truly unique (though unlikely with this simple method in a real scenario)
    while profile_id in existing_profile_ids:
        # This part is a safeguard; for a robust system, a better ID generation is needed (e.g., UUID)
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        profile_id = f"user_{len(existing_profile_ids) + 1}_{random_suffix}"

    profile_data['profile_id'] = profile_id
    existing_profile_ids.append(profile_id)
    return profile_data

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
