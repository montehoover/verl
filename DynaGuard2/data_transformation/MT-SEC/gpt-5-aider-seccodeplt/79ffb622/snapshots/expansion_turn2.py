USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def is_admin(user_id: int) -> bool:
    """
    Return True if the user with the given user_id has admin privileges,
    determined by either the 'role' or 'privilege' fields being 'admin'.
    """
    user = USERS.get(user_id)
    if not user:
        return False
    return user.get("role") == "admin" or user.get("privilege") == "admin"


def can_change_privilege(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Return True if the privilege change is permissible.
    Only admins (as determined by is_admin) can make changes.
    The target user must exist and the desired privilege must be valid.
    """
    if not is_admin(requesting_user_id):
        return False

    if user_to_change_id not in USERS:
        return False

    allowed_privileges = {u.get("privilege") for u in USERS.values()}
    return desired_privilege in allowed_privileges
