def is_user_admin(user_id: int) -> bool:
    """
    Check if the user with the given user_id has the 'admin' role.

    Expects a global data structure (e.g., USER_ROLES) that maps user IDs to their roles.
    The roles can be:
      - a string (e.g., "admin")
      - an iterable of strings (e.g., ["admin", "editor"] or {"admin", "user"})
      - a dict containing a 'roles' key with one of the above forms

    Returns:
        True if the user is an admin, otherwise False.
    """
    if not isinstance(user_id, int):
        return False

    # Access the global roles structure if available.
    try:
        roles_data = USER_ROLES  # type: ignore[name-defined]
    except NameError:
        return False

    # Fetch this user's roles from the global data.
    roles = None
    try:
        roles = roles_data.get(user_id)  # type: ignore[attr-defined]
    except AttributeError:
        # If the global structure is not a mapping with .get, we can't proceed safely.
        return False

    if roles is None:
        return False

    # If roles is a mapping with a 'roles' key, drill into it.
    if isinstance(roles, dict) and "roles" in roles:
        roles = roles["roles"]

    # If roles is a string, compare directly.
    if isinstance(roles, str):
        return roles.strip().lower() == "admin"

    # If roles is iterable, check membership case-insensitively.
    try:
        iterator = iter(roles)  # Ensure it's iterable
    except TypeError:
        return False

    for r in iterator:
        if isinstance(r, str) and r.strip().lower() == "admin":
            return True

    return False
