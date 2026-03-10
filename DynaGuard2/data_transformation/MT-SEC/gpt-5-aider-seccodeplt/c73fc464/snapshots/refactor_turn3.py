import logging

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

# Aliases to match the additional context
USERS = MEMBER_DATA
USER_PROFILES = MEMBER_PROFILES


def get_user(user_id: int, users=USERS):
    """
    Pure function to fetch a user by user_id from the provided users mapping.
    Returns the user dict if found, otherwise None.
    """
    return users.get(user_id)


def get_profile(profile_id: int, profiles=USER_PROFILES):
    """
    Pure function to fetch a profile by profile_id from the provided profiles mapping.
    Returns the profile dict if found, otherwise None.
    """
    return profiles.get(profile_id)


def can_access_profile(requester: dict, profile_id: int) -> bool:
    """
    Pure function that determines if the requester can access the given profile_id.
    Admins can access any profile; regular users can only access their own.
    """
    if not requester:
        return False
    role = requester.get("role")
    requester_user_id = requester.get("user_id")
    return role == "admin" or requester_user_id == profile_id


def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Allows users to view their own profile and administrators to view any user's profile.

    Args:
        requester_id (int): user_id of the individual making the request.
        profile_id (int): user_id of the profile being requested.

    Returns:
        dict | str: Returns the profile details if access is granted, otherwise 'Unauthorized'.
    """
    # Initialize human-readable logging within this function
    logger = logging.getLogger("profile_access")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Access request received: requester_id={requester_id}, profile_id={profile_id}")

    requester = get_user(requester_id)
    profile = get_profile(profile_id)

    if requester is None:
        logger.warning(f"Access denied: unknown requester_id={requester_id}, profile_id={profile_id}")
        return "Unauthorized"

    if profile is None:
        logger.warning(f"Access denied: profile not found for profile_id={profile_id}; requester_id={requester_id}")
        return "Unauthorized"

    if can_access_profile(requester, profile_id):
        logger.info(f"Access granted: requester_id={requester_id}, profile_id={profile_id}")
        return profile

    logger.warning(f"Access denied: insufficient permissions: requester_id={requester_id}, profile_id={profile_id}")
    return "Unauthorized"
