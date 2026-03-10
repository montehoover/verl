USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def is_admin(user_id: int) -> bool:
    """
    Return True if the user with the given user_id has admin privileges, else False.
    """
    user = USERS.get(user_id)
    if not user:
        return False

    role = str(user.get("role", "")).lower()
    privilege = str(user.get("privilege", "")).lower()
    return role == "admin" or privilege == "admin"


def can_change_privilege(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Return True if current_user_id is allowed to change target_user_id's privilege to new_privilege.
    Requirements:
      - current_user_id must be an admin (checked via is_admin).
      - target_user_id must exist.
      - new_privilege must be one of the allowed privilege values.
    """
    if not is_admin(current_user_id):
        return False

    if target_user_id not in USERS:
        return False

    allowed_privileges = {"admin", "moderator", "normal"}
    try:
        desired = str(new_privilege).lower()
    except Exception:
        return False

    return desired in allowed_privileges
