import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def _is_admin(user_id: int) -> bool:
    """Checks if a user has the 'admin' role."""
    if user_id in USERS:
        return USERS[user_id].get("role") == "admin"
    return False


def _get_user_profile(user_id: int):
    """Retrieves a user's profile if it exists."""
    return USER_PROFILES.get(user_id, "Unauthorized")


def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Grants users access to their own profile, or enables administrators
    to view any given user's profile.

    Args:
        requesting_user_id: int, representing the ID of the user making the request.
        viewing_user_id: int, representing the ID of the profile being requested.

    Returns:
        The profile of the specified user if access is allowed.
        Otherwise, it should return 'Unauthorized'.
    """
    logger.info(f"Access attempt: User {requesting_user_id} requesting profile of user {viewing_user_id}.")

    if requesting_user_id not in USERS:
        logger.warning(
            f"Outcome: Unauthorized. Requesting user {requesting_user_id} does not exist."
        )
        return "Unauthorized"

    # User can access if they are viewing their own profile or if they are an admin
    if requesting_user_id == viewing_user_id or _is_admin(requesting_user_id):
        profile = _get_user_profile(viewing_user_id)
        if profile == "Unauthorized":  # Profile to be viewed does not exist
            logger.warning(
                f"Outcome: Unauthorized. User {requesting_user_id} permitted to access profile "
                f"of user {viewing_user_id}, but profile for user {viewing_user_id} does not exist."
            )
        else:
            logger.info(
                f"Outcome: Authorized. User {requesting_user_id} accessed profile of user {viewing_user_id}."
            )
        return profile
    else:
        logger.warning(
            f"Outcome: Unauthorized. User {requesting_user_id} not permitted to access profile of user {viewing_user_id}."
        )
        return "Unauthorized"
