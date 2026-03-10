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

def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Lets users view their own profile, and gives administrators the ability 
    to view any user's profile.

    Args:
        initiator_id: int, user_id of the individual making the request.
        requested_profile_id: int, user_id of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized' 
        if the request is invalid.
    """
    initiator_user = SYSTEM_USERS.get(initiator_id)
    if not initiator_user:
        return "Unauthorized"

    requested_profile = USER_PROFILES.get(requested_profile_id)
    if not requested_profile:
        return "Unauthorized"

    initiator_role = initiator_user.get("role")

    if initiator_role == "admin":
        return requested_profile
    elif initiator_role == "user":
        if initiator_id == requested_profile_id:
            return requested_profile
        else:
            return "Unauthorized"
    else:
        # Handles cases where role might be missing or is not 'admin' or 'user'
        return "Unauthorized"
