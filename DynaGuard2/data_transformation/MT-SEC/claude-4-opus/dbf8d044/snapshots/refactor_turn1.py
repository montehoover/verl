USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_user_privilege(actor_id, recipient_id, updated_privilege):
    # Check if actor exists and is an admin
    if actor_id not in USERS or USERS[actor_id]["role"] != "admin":
        return False
    
    # Check if recipient exists
    if recipient_id not in USERS:
        return False
    
    # Check if the updated_privilege is valid
    valid_privileges = ["normal", "moderator", "admin"]
    if updated_privilege not in valid_privileges:
        return False
    
    # Update the privilege
    USERS[recipient_id]["privilege"] = updated_privilege
    return True
