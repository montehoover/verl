from collections.abc import Mapping, Iterable

def is_user_admin(user_id: int) -> bool:
    """
    Return True if the given user_id has an admin role according to the global USER_ROLES.

    Expected global:
      USER_ROLES: Mapping[int, Union[str, Iterable[str], Mapping]]
        - If str: e.g., "admin"
        - If iterable of str: e.g., ["member", "admin"]
        - If mapping: e.g., {"role": "admin"}, {"roles": ["admin"]}, or {"admin": True}
    """
    if not isinstance(user_id, int):
        return False

    roles_ds = globals().get("USER_ROLES")
    if not isinstance(roles_ds, Mapping):
        return False

    roles = roles_ds.get(user_id)
    if roles is None:
        return False

    if isinstance(roles, str):
        return roles.strip().lower() == "admin"

    if isinstance(roles, Mapping):
        if roles.get("admin") is True:
            return True
        value = roles.get("role") or roles.get("roles") or roles.get("type")
        if isinstance(value, str):
            return value.strip().lower() == "admin"
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return any(isinstance(r, str) and r.strip().lower() == "admin" for r in value)
        if "admin" in roles and bool(roles["admin"]):
            return True
        return False

    if isinstance(roles, Iterable) and not isinstance(roles, (str, bytes)):
        return any(isinstance(r, str) and r.strip().lower() == "admin" for r in roles)

    return False
