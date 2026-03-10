def is_user_admin(user_id: int) -> bool:
    """
    Determine whether the user with the given user_id is an admin.

    Expects a global data structure named USER_ROLES that tracks user roles.
    Supported structures:
      - dict[int, str] where the value is a role name, e.g., "admin"
      - dict[int, Iterable[str]] where one of the roles is "admin"
      - dict[int, dict] where nested keys include:
          - "is_admin": bool
          - or "roles"/"role": str or Iterable[str]
      - set/list/tuple of admin user_ids

    Returns:
        True if the user is an admin, otherwise False.
    """
    roles_data = globals().get("USER_ROLES")
    if roles_data is None:
        return False

    # Mapping-like: user_id -> role info
    if hasattr(roles_data, "get"):
        try:
            user_entry = roles_data.get(user_id)  # type: ignore[attr-defined]
        except Exception:
            user_entry = None

        if user_entry is None:
            return False

        # If explicitly a boolean flag
        if isinstance(user_entry, bool):
            return bool(user_entry)

        # If nested structure
        if isinstance(user_entry, dict):
            if "is_admin" in user_entry:
                return bool(user_entry["is_admin"])
            roles = user_entry.get("roles")
            if roles is None:
                roles = user_entry.get("role")
            if roles is None:
                return False
            user_entry = roles

        # If single role string
        if isinstance(user_entry, str):
            return user_entry.strip().lower() == "admin"

        # If iterable of roles
        try:
            for r in user_entry:
                if isinstance(r, str) and r.strip().lower() == "admin":
                    return True
            return False
        except TypeError:
            return False

    # Collection of admin user IDs
    if isinstance(roles_data, (set, list, tuple)):
        return user_id in roles_data

    return False
