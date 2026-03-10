USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Allows an admin user to update the privilege level of another user.

    Args:
        actor_id (int): The ID of the user making the request (must be an admin).
        recipient_id (int): The ID of the user whose privilege is being changed.
        updated_privilege (str): Desired privilege: 'normal', 'moderator', or 'admin'.

    Returns:
        bool: True if the privilege change is (or is already) in effect, otherwise False.
    """
    # Validate users exist
    actor = USERS.get(actor_id)
    recipient = USERS.get(recipient_id)
    if actor is None or recipient is None:
        return False

    # Normalize and validate target privilege
    if not isinstance(updated_privilege, str):
        return False
    target = updated_privilege.strip().lower()
    allowed = {"normal", "moderator", "admin"}
    if target not in allowed:
        return False

    # Only admins can modify privileges
    if actor.get("role") != "admin":
        return False

    # Must modify other user accounts (no self-modification)
    if actor_id == recipient_id:
        return False

    # Apply change (idempotent: return True even if already set)
    if recipient.get("privilege") == target:
        return True

    recipient["privilege"] = target
    return True
