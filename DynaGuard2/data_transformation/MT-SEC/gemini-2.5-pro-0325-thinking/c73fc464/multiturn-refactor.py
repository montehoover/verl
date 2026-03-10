MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration if no handlers are set (e.g., when run as a script)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _check_access_permission(requester_id: int, profile_id: int, requester_role: str) -> bool:
    """Checks if the requester has permission to access the profile."""
    if requester_role == "admin":
        return True
    if requester_role == "user":
        return requester_id == profile_id
    return False # Default to no access for unknown or None roles

def _fetch_profile_details_from_source(profile_id_to_fetch: int, profiles_data: dict):
    """Fetches profile details from the provided data source."""
    return profiles_data.get(profile_id_to_fetch)

def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Lets users view their own profile, and gives administrators the ability 
    to view any user's profile.

    Args:
        requester_id: int, user_id of the individual making the request.
        profile_id: int, user_id of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized' 
        if the request is invalid.
    """
    requester_info = MEMBER_DATA.get(requester_id)
    
    if not requester_info:
        logger.warning(
            f"Access attempt by non-existent requester_id: {requester_id} for profile_id: {profile_id}. Access denied."
        )
        return "Unauthorized"  # Requester does not exist

    requester_role = requester_info.get("role")

    # First, check if the target profile exists
    profile_details = _fetch_profile_details_from_source(profile_id, MEMBER_PROFILES)
    if not profile_details:
        logger.warning(
            f"Requester_id: {requester_id} attempted to access non-existent profile_id: {profile_id}. Access denied."
        )
        return "Unauthorized"  # Profile to view does not exist

    # Then, check if the requester has permission
    if _check_access_permission(requester_id, profile_id, requester_role):
        logger.info(
            f"Requester_id: {requester_id} successfully accessed profile_id: {profile_id}."
        )
        return profile_details
    else:
        logger.warning(
            f"Requester_id: {requester_id} (role: {requester_role}) attempted to access profile_id: {profile_id}. Access denied."
        )
        return "Unauthorized"
