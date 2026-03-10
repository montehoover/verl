def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Returns the profile details if access is granted, otherwise 'Unauthorized'.

    Rules:
    - A user can view their own profile.
    - An admin can view any user's profile.
    """
    users = globals().get('USERS') or globals().get('SYSTEM_USERS') or {}
    profiles = globals().get('USER_PROFILES') or {}

    initiator = users.get(initiator_id)
    if not initiator:
        return 'Unauthorized'

    is_admin = initiator.get('role') == 'admin'
    is_self = initiator_id == requested_profile_id

    if not (is_admin or is_self):
        return 'Unauthorized'

    profile = profiles.get(requested_profile_id)
    if not profile:
        return 'Unauthorized'

    # Return a copy to prevent accidental mutation of the global state.
    return dict(profile)
