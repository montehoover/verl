import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SYSTEM_USERS = {
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


def _get_user_role(user_id: int):
    """Helper function to get a user's role."""
    user = SYSTEM_USERS.get(user_id)
    if user:
        return user.get("role")
    return None


def _get_user_profile(user_id: int):
    """Helper function to get a user's profile."""
    return USER_PROFILES.get(user_id)


def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Lets users view their own profile, and gives administrators the ability to view any user's profile.

    Args:
        initiator_id: int, user_id of the individual making the request.
        requested_profile_id: int, user_id of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized' if the request is invalid.
    """
    logging.info(f"Access attempt: initiator_id={initiator_id}, requested_profile_id={requested_profile_id}")

    # Check if initiator and requested profile exist
    if initiator_id not in SYSTEM_USERS:
        logging.warning(f"Unauthorized: Initiator ID {initiator_id} not found.")
        return "Unauthorized"
    
    if requested_profile_id not in USER_PROFILES:
        logging.warning(f"Unauthorized: Requested profile ID {requested_profile_id} not found.")
        return "Unauthorized"

    initiator_role = _get_user_role(initiator_id)
    requested_profile_data = _get_user_profile(requested_profile_id)

    # User requesting their own profile
    if initiator_id == requested_profile_id:
        logging.info(f"Access granted: User {initiator_id} accessing own profile {requested_profile_id}.")
        return requested_profile_data

    # Admin requesting any profile
    if initiator_role == "admin":
        logging.info(f"Access granted: Admin {initiator_id} accessing profile {requested_profile_id}.")
        return requested_profile_data

    logging.warning(f"Access denied: User {initiator_id} (role: {initiator_role}) attempted to access profile {requested_profile_id}.")
    return "Unauthorized"
