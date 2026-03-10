USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id):
    """Check if a user exists and has admin role."""
    return user_id in USERS and USERS[user_id]["role"] == "admin"

def user_exists(user_id):
    """Check if a user exists in the system."""
    return user_id in USERS

def is_valid_privilege(privilege):
    """Check if the privilege level is valid."""
    valid_privileges = ["normal", "moderator", "admin"]
    return privilege in valid_privileges

def set_user_privilege(user_id, privilege):
    """Update the privilege level for a user."""
    USERS[user_id]["privilege"] = privilege

def update_user_privilege(actor_id, recipient_id, updated_privilege):
    # Pipeline of validation checks
    if not is_admin(actor_id):
        return False
    
    if not user_exists(recipient_id):
        return False
    
    if not is_valid_privilege(updated_privilege):
        return False
    
    # Update privilege if all checks pass
    set_user_privilege(recipient_id, updated_privilege)
    return True
