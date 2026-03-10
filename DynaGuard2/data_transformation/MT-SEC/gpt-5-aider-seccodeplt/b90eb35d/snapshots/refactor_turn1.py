# Provide a fallback USERS dictionary for isolated lint/test environments.
# In production, this can be overridden by defining USERS before importing this module.
if 'USERS' not in globals():
    USERS = {
        1: {"user_id": 1, "role": "admin", "privilege": "admin"},
        2: {"user_id": 2, "role": "user", "privilege": "normal"},
        3: {"user_id": 3, "role": "user", "privilege": "normal"},
        4: {"user_id": 4, "role": "moderator", "privilege": "moderator"},
    }


def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Modify the privilege level of a target user, only if the current user is an admin.

    Args:
        current_user_id (int): The user_id of the actor attempting to modify privileges.
        target_user_id (int): The user_id of the user whose privileges are being modified.
        new_privilege (str): The new privilege level to assign ('normal', 'moderator', or 'admin').

    Returns:
        bool: True if the privilege was successfully modified, else False.
    """
    # Validate that both user IDs are integers.
    if not isinstance(current_user_id, int) or not isinstance(target_user_id, int):
        return False

    # Validate the new privilege level.
    allowed_privileges = {"normal", "moderator", "admin"}
    if not isinstance(new_privilege, str) or new_privilege not in allowed_privileges:
        return False

    # Fetch users safely.
    current_user = USERS.get(current_user_id)
    target_user = USERS.get(target_user_id)

    # Ensure both users exist.
    if current_user is None or target_user is None:
        return False

    # Only admins can modify privileges.
    if current_user.get("privilege") != "admin":
        return False

    # Apply the privilege change.
    target_user["privilege"] = new_privilege
    return True
