import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}


def _is_user_admin(user_id: int, users_data: dict) -> bool:
    """Checks if a user has an admin role."""
    if user_id not in users_data:
        return False
    return users_data[user_id].get("role") == "admin"


def _get_user_profile_data(user_id: int, profiles_data: dict) -> dict | None:
    """Retrieves a user profile if it exists."""
    return profiles_data.get(user_id)


def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Grants users access to their own profile, or enables administrators 
    to view any given user's profile.

    Args:
        active_user_id: int, representing the ID of the user making the request.
        profile_user_id: int, representing the ID of the profile being requested.

    Returns:
        The profile of the specified user if access is allowed.
        Otherwise, it returns 'Unauthorized'.
    """
    logging.info(f"Access request: active_user_id={active_user_id}, profile_user_id={profile_user_id}")

    if active_user_id not in USERS:
        logging.warning(f"Unauthorized: Active user {active_user_id} not found.")
        return "Unauthorized"

    profile_to_view = _get_user_profile_data(profile_user_id, USER_PROFILES)

    if not profile_to_view:
        logging.warning(f"Unauthorized: Profile for user {profile_user_id} not found.")
        return "Unauthorized" # Or indicate profile not found

    is_admin = _is_user_admin(active_user_id, USERS)
    
    if is_admin or active_user_id == profile_user_id:
        logging.info(f"Access granted: active_user_id={active_user_id} to profile_user_id={profile_user_id}")
        return profile_to_view
    
    logging.warning(f"Unauthorized: Access denied for active_user_id={active_user_id} to profile_user_id={profile_user_id}")
    return "Unauthorized"
