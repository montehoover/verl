USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.

    Args:
        actor_id: The ID of the user making the request.
        recipient_id: The ID of the user whose privilege is being changed.
        updated_privilege: The desired privilege level for the target user.

    Returns:
        True if the privilege change takes effect successfully, otherwise False.
    """
    actor = USERS.get(actor_id)
    recipient = USERS.get(recipient_id)

    if not actor or actor.get("privilege") != "admin":
        return False  # Actor is not an admin or does not exist

    if not recipient:
        return False  # Recipient does not exist

    if updated_privilege not in VALID_PRIVILEGES:
        return False # Invalid privilege level

    # Update privilege
    USERS[recipient_id]["privilege"] = updated_privilege
    return True
