def get_all_admin_users() -> list[int]:
    """
    Return a list of user IDs (ints) for all users who are admins according to the global USER_ROLES mapping.

    USER_ROLES is expected to map integer user IDs to role info, which may be:
    - a string role, e.g., "admin"
    - an iterable of roles, e.g., ["user", "admin"]
    - a dict that either has a boolean 'admin' key or a 'roles' key containing roles

    If USER_ROLES is not defined or malformed, returns an empty list.
    """
    try:
        roles_map = USER_ROLES  # type: ignore[name-defined]
    except NameError:
        return []

    if not isinstance(roles_map, dict):
        return []

    admin_ids: list[int] = []
    for user_id, roles_info in roles_map.items():
        if not isinstance(user_id, int):
            continue

        is_admin = False

        if isinstance(roles_info, str):
            is_admin = roles_info.lower() == "admin"

        elif isinstance(roles_info, dict):
            if "admin" in roles_info:
                is_admin = bool(roles_info["admin"])
            else:
                roles = roles_info.get("roles")
                if isinstance(roles, str):
                    is_admin = roles.lower() == "admin"
                elif isinstance(roles, (list, tuple, set)):
                    is_admin = any(isinstance(r, str) and r.lower() == "admin" for r in roles)

        elif isinstance(roles_info, (list, tuple, set)):
            is_admin = any(isinstance(r, str) and r.lower() == "admin" for r in roles_info)

        if is_admin:
            admin_ids.append(user_id)

    return admin_ids
