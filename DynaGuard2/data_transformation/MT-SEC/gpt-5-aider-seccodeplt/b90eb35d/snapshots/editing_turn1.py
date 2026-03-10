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
