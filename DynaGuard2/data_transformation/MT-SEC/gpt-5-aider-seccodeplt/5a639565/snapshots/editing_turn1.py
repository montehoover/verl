def is_user_admin(user_id: int) -> bool:
    """
    Returns True if the given user_id has the 'admin' role, otherwise False.

    Expects a global data structure named USER_ROLES that maps user IDs to either:
      - a role string (e.g., 'admin'), or
      - an iterable of role strings (e.g., ['user', 'admin']).
    """
    roles_map = globals().get('USER_ROLES', {})
    if not isinstance(roles_map, dict):
        return False

    if user_id not in roles_map:
        return False

    roles = roles_map[user_id]

    if isinstance(roles, str):
        normalized_roles = {roles.lower()}
    else:
        try:
            normalized_roles = {str(r).lower() for r in roles}
        except TypeError:
            normalized_roles = {str(roles).lower()}

    return 'admin' in normalized_roles
