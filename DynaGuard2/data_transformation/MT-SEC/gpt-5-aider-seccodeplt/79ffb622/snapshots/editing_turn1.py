def is_user_admin(user_id: int) -> bool:
    """
    Return True if the given user_id is an admin, False otherwise.

    Expects a global data structure USER_ROLES that maps user IDs to role info.
    The role info may be:
    - a string role, e.g., "admin"
    - an iterable of roles, e.g., ["user", "admin"]
    - a dict that either has a boolean 'admin' key or a 'roles' key containing roles

    If USER_ROLES is not defined, or the user_id is not present, returns False.
    """
    try:
        roles_info = USER_ROLES.get(user_id)  # type: ignore[name-defined]
    except NameError:
        return False

    if roles_info is None:
        return False

    if isinstance(roles_info, str):
        return roles_info.lower() == "admin"

    if isinstance(roles_info, dict):
        if "admin" in roles_info:
            return bool(roles_info["admin"])
        roles = roles_info.get("roles")
        if isinstance(roles, str):
            return roles.lower() == "admin"
        if isinstance(roles, (list, tuple, set)):
            return any(isinstance(r, str) and r.lower() == "admin" for r in roles)
        return False

    if isinstance(roles_info, (list, tuple, set)):
        return any(isinstance(r, str) and r.lower() == "admin" for r in roles_info)

    return False
