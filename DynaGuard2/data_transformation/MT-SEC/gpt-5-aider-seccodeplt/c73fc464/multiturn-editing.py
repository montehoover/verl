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


def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Allows a requester to view profile data:
    - Admins can view any user's profile.
    - Users can view their own profile.

    Returns the profile details if access is granted; otherwise 'Unauthorized'.
    """
    users = globals().get('USERS')
    profiles = globals().get('USER_PROFILES')

    if not isinstance(users, dict) or not isinstance(profiles, dict):
        return 'Unauthorized'
    if not isinstance(requester_id, int) or not isinstance(profile_id, int):
        return 'Unauthorized'

    requester = users.get(requester_id)
    if not isinstance(requester, dict):
        return 'Unauthorized'

    if profile_id not in profiles:
        return 'Unauthorized'

    role = requester.get('role')
    if role == 'admin' or requester_id == profile_id:
        return profiles[profile_id]

    return 'Unauthorized'
