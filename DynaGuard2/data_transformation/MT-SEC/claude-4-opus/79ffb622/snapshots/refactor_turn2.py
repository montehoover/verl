USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_valid_privilege(privilege: str) -> bool:
    """Check if the given privilege is valid."""
    valid_privileges = ["normal", "moderator", "admin"]
    return privilege in valid_privileges

def user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in USERS

def is_admin(user_id: int) -> bool:
    """Check if a user has admin privileges."""
    return user_id in USERS and USERS[user_id]["privilege"] == "admin"

def update_user_privilege(user_id: int, new_privilege: str) -> None:
    """Update a user's privilege level."""
    USERS[user_id]["privilege"] = new_privilege

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    # Check if requesting user exists and is an admin
    if not user_exists(requesting_user_id):
        return False
    
    if not is_admin(requesting_user_id):
        return False
    
    # Check if user to change exists
    if not user_exists(user_to_change_id):
        return False
    
    # Check if desired privilege is valid
    if not is_valid_privilege(desired_privilege):
        return False
    
    # Change the privilege
    update_user_privilege(user_to_change_id, desired_privilege)
    
    return True
