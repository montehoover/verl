USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    # Check if acting user exists and has admin privileges
    if acting_user_id not in USERS:
        return False
    
    if USERS[acting_user_id]["privilege"] != "admin":
        return False
    
    # Check if affected user exists
    if affected_user_id not in USERS:
        return False
    
    # Check if new_role is valid
    valid_roles = ["normal", "moderator", "admin"]
    if new_role not in valid_roles:
        return False
    
    # Update the affected user's privilege
    USERS[affected_user_id]["privilege"] = new_role
    
    # Update the role based on privilege
    if new_role == "admin":
        USERS[affected_user_id]["role"] = "admin"
    elif new_role == "moderator":
        USERS[affected_user_id]["role"] = "moderator"
    else:
        USERS[affected_user_id]["role"] = "user"
    
    return True
