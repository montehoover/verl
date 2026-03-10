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
        actor_id: int, the ID of the user making the request
        recipient_id: int, the ID of the user whose privilege is being changed
        updated_privilege: str, the desired privilege level for the target user
    
    Returns:
        bool: True if the privilege change takes effect successfully, otherwise False
    """
    # Check if actor exists and has admin privilege
    if actor_id not in USERS:
        return False
    
    actor = USERS[actor_id]
    if actor.get("privilege") != "admin":
        return False
    
    # Check if recipient exists
    if recipient_id not in USERS:
        return False
    
    # Validate the updated privilege level
    valid_privileges = {"normal", "moderator", "admin"}
    if updated_privilege not in valid_privileges:
        return False
    
    # Update the privilege
    USERS[recipient_id]["privilege"] = updated_privilege
    return True
