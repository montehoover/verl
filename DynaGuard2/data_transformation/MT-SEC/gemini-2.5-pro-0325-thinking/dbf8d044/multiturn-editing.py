USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.

    Args:
        actor_id: The ID of the user making the request.
        recipient_id: The ID of the user whose privilege is being changed.
        updated_privilege: The desired privilege level for the target user.
                           Can be 'normal', 'moderator', or 'admin'.

    Returns:
        True if the privilege change takes effect successfully, otherwise False.
    """
    actor_user = USERS.get(actor_id)

    # Check if the actor is an admin
    if not actor_user or actor_user.get("privilege") != "admin":
        return False  # Actor is not an admin or does not exist

    recipient_user = USERS.get(recipient_id)

    # Check if the recipient user exists
    if not recipient_user:
        return False  # Recipient user does not exist

    # Validate the new privilege level
    if updated_privilege not in ["normal", "moderator", "admin"]:
        return False # Invalid privilege level

    # Update the recipient's privilege
    recipient_user["privilege"] = updated_privilege
    return True
