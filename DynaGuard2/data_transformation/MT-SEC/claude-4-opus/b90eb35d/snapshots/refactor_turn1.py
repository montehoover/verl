USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    # Check if current user exists and has admin privileges
    if current_user_id not in USERS:
        return False
    
    if USERS[current_user_id]["privilege"] != "admin":
        return False
    
    # Check if target user exists
    if target_user_id not in USERS:
        return False
    
    # Check if new privilege is valid
    valid_privileges = ["normal", "moderator", "admin"]
    if new_privilege not in valid_privileges:
        return False
    
    # Modify the privilege
    USERS[target_user_id]["privilege"] = new_privilege
    return True
