def view_own_profile(user_id: int):
    """
    Allows a user to view their own profile information.

    Looks up the profile in the global USER_PROFILES mapping by user_id.
    Returns the profile details if found; otherwise returns 'User not found'.
    """
    profiles = globals().get('USER_PROFILES')
    if isinstance(profiles, dict) and isinstance(user_id, int):
        return profiles.get(user_id, 'User not found')
    return 'User not found'
