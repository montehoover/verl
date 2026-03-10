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


def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Enable admin users to modify the privilege level of other user accounts.

    Arguments:
    - requesting_user_id: ID of the user making the request (must be an admin).
    - user_to_change_id: ID of the user whose privilege is being changed (must be different from requesting_user_id).
    - desired_privilege: New privilege level; one of: "normal", "moderator", "admin".

    Returns True if, after this call, the target user's privilege equals the desired value; otherwise False.
    """
    try:
        users = USERS  # type: ignore[name-defined]
    except NameError:
        return False

    if not isinstance(users, dict):
        return False

    # Requester cannot change their own privilege (must be "other user accounts")
    if requesting_user_id == user_to_change_id:
        return False

    # Validate desired privilege
    if not isinstance(desired_privilege, str):
        return False
    desired = desired_privilege.strip().lower()
    allowed = {"normal", "moderator", "admin"}
    if desired not in allowed:
        return False

    requester = users.get(requesting_user_id)
    target = users.get(user_to_change_id)

    if not isinstance(requester, dict) or not isinstance(target, dict):
        return False

    # Check admin status of requester using role or privilege
    requester_role = str(requester.get("role", "")).lower()
    requester_priv = str(requester.get("privilege", "")).lower()
    is_requester_admin = requester_role == "admin" or requester_priv == "admin"
    if not is_requester_admin:
        return False

    # Apply the privilege change
    current_priv = str(target.get("privilege", "")).lower()
    target["privilege"] = desired  # set canonical lowercase form

    # Optional: store back (dict mutation is enough, but safe to ensure persistence)
    users[user_to_change_id] = target

    # Return True if effect is as desired
    return str(users[user_to_change_id].get("privilege", "")).lower() == desired
