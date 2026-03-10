def get_user_privilege(user_id: int) -> str:
    """
    Return the privilege level for the given user_id.

    The function will use any of the following global data structures (first match wins):
    - USER_PRIVILEGES: Dict[int, str]
    - USER_ROLES and ROLE_PRIVILEGES:
        USER_ROLES: Dict[int, str]           # user_id -> role
        ROLE_PRIVILEGES: Dict[str, str]      # role -> privilege
    - USERS: Dict[int, dict] where the user dict may contain:
        - 'privilege': str
        - or 'role': str (resolved via ROLE_PRIVILEGES)
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    g = globals()

    # Case 1: Direct mapping from user_id to privilege string
    if 'USER_PRIVILEGES' in g and isinstance(g['USER_PRIVILEGES'], dict):
        try:
            privilege = g['USER_PRIVILEGES'][user_id]
        except KeyError:
            raise KeyError(f"user_id {user_id} not found in USER_PRIVILEGES")
        if not isinstance(privilege, str):
            raise TypeError("Privilege must be a string")
        return privilege

    # Case 2: Resolve via role maps
    if (
        'USER_ROLES' in g and isinstance(g['USER_ROLES'], dict) and
        'ROLE_PRIVILEGES' in g and isinstance(g['ROLE_PRIVILEGES'], dict)
    ):
        try:
            role = g['USER_ROLES'][user_id]
        except KeyError:
            raise KeyError(f"user_id {user_id} not found in USER_ROLES")
        try:
            privilege = g['ROLE_PRIVILEGES'][role]
        except KeyError:
            raise KeyError(f"role {role!r} not found in ROLE_PRIVILEGES")
        if not isinstance(privilege, str):
            raise TypeError("Privilege must be a string")
        return privilege

    # Case 3: USERS dict with embedded privilege or role
    if 'USERS' in g and isinstance(g['USERS'], dict):
        try:
            user = g['USERS'][user_id]
        except KeyError:
            raise KeyError(f"user_id {user_id} not found in USERS")
        if not isinstance(user, dict):
            raise TypeError("USERS must map user_id to dicts")
        if 'privilege' in user:
            privilege = user['privilege']
            if not isinstance(privilege, str):
                raise TypeError("User privilege must be a string")
            return privilege
        if 'role' in user:
            role = user['role']
            if 'ROLE_PRIVILEGES' in g and isinstance(g['ROLE_PRIVILEGES'], dict):
                try:
                    privilege = g['ROLE_PRIVILEGES'][role]
                except KeyError:
                    raise KeyError(f"role {role!r} not found in ROLE_PRIVILEGES")
                if not isinstance(privilege, str):
                    raise TypeError("Privilege must be a string")
                return privilege
            raise RuntimeError("ROLE_PRIVILEGES global not available to resolve user's role to privilege")
        raise KeyError(f"User record for {user_id} lacks 'privilege' or 'role' field")

    raise RuntimeError(
        "No recognized global data structure found. Expected one of: "
        "USER_PRIVILEGES, USER_ROLES+ROLE_PRIVILEGES, or USERS"
    )


def generate_user_report() -> list[dict]:
    """
    Generate a report of all users and their privilege levels from the USERS dictionary.

    Returns:
        List of dicts in the form: [{'user_id': int, 'privilege': str}, ...]
    """
    g = globals()
    if 'USERS' not in g or not isinstance(g['USERS'], dict):
        raise RuntimeError("USERS global not available for report generation")

    report = []
    for user_id, user in g['USERS'].items():
        if not isinstance(user_id, int):
            raise TypeError("USERS keys must be integers")
        if isinstance(user, dict) and 'privilege' in user and isinstance(user['privilege'], str):
            privilege = user['privilege']
        else:
            privilege = get_user_privilege(user_id)
            if not isinstance(privilege, str):
                raise TypeError("Privilege must be a string")
        report.append({'user_id': user_id, 'privilege': privilege})

    return report


def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Modify the privilege of target_user_id to new_privilege, only if current_user_id is an admin.

    Args:
        current_user_id: The user attempting the change.
        target_user_id: The user whose privilege is being modified.
        new_privilege: The privilege level to assign ('normal', 'moderator', 'admin').

    Returns:
        True if the privilege was successfully modified, else False.
    """
    g = globals()
    users = g.get('USERS')
    if not isinstance(users, dict):
        return False

    if not isinstance(current_user_id, int) or not isinstance(target_user_id, int) or not isinstance(new_privilege, str):
        return False

    new_privilege = new_privilege.strip()
    allowed_privileges = {'normal', 'moderator', 'admin'}
    if new_privilege not in allowed_privileges:
        return False

    current_user = users.get(current_user_id)
    if not isinstance(current_user, dict):
        return False

    try:
        current_privilege = get_user_privilege(current_user_id)
    except Exception:
        current_privilege = current_user.get('privilege') if isinstance(current_user, dict) else None

    if current_privilege != 'admin':
        return False

    target_user = users.get(target_user_id)
    if not isinstance(target_user, dict):
        return False

    # Perform the update
    target_user['privilege'] = new_privilege
    return True
