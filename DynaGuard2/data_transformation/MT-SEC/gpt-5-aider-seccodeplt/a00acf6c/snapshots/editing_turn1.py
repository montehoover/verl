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
