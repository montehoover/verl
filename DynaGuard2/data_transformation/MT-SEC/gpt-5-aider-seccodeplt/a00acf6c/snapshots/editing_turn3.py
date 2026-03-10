def view_own_profile(user_id: int):
    """
    Return the profile details for the given user_id from the global USER_PROFILES.
    If no profile exists, return 'Profile not found'.
    """
    global USER_PROFILES
    try:
        profile = USER_PROFILES.get(user_id)
    except NameError:
        return 'Profile not found'
    return profile if profile is not None else 'Profile not found'


def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Allows users to view their own profile and administrators to view any user's profile.
    Returns the profile details if access is granted; otherwise 'Unauthorized'.
    """
    # Resolve users dictionary (supports USERS or SYSTEM_USERS for compatibility)
    try:
        users = USERS  # type: ignore[name-defined]
    except NameError:
        try:
            users = SYSTEM_USERS  # type: ignore[name-defined]
        except NameError:
            return 'Unauthorized'

    try:
        profiles = USER_PROFILES  # type: ignore[name-defined]
    except NameError:
        return 'Unauthorized'

    initiator = users.get(initiator_id)
    if not initiator:
        return 'Unauthorized'

    role = initiator.get('role')
    if initiator_id != requested_profile_id and role != 'admin':
        return 'Unauthorized'

    profile = profiles.get(requested_profile_id)
    if profile is None:
        return 'Unauthorized'

    return profile
