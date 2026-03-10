USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Update the privilege of a recipient user if the actor is an admin modifying another user's privilege.

    Args:
        actor_id (int): ID of the user performing the action.
        recipient_id (int): ID of the user whose privilege is being updated.
        updated_privilege (str): Target privilege level ('normal', 'moderator', or 'admin').

    Returns:
        bool: True if the update succeeds; otherwise False.
    """
    # Validate inputs and existence
    actor = USERS.get(actor_id)
    recipient = USERS.get(recipient_id)
    if actor is None or recipient is None:
        return False

    # Only admin actors can modify privileges
    if actor.get("privilege") != "admin":
        return False

    # Only allow modifying other users' privileges (not self)
    if actor_id == recipient_id:
        return False

    # Normalize and validate target privilege
    if not isinstance(updated_privilege, str):
        return False
    target_priv = updated_privilege.strip().lower()
    if target_priv not in ALLOWED_PRIVILEGES:
        return False

    # Apply change
    recipient["privilege"] = target_priv
    return True
