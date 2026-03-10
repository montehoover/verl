USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    # Check if requesting user exists and is an admin
    if requesting_user_id not in USERS:
        return False
    
    if USERS[requesting_user_id]["privilege"] != "admin":
        return False
    
    # Check if user to change exists
    if user_to_change_id not in USERS:
        return False
    
    # Check if desired privilege is valid
    valid_privileges = ["normal", "moderator", "admin"]
    if desired_privilege not in valid_privileges:
        return False
    
    # Change the privilege
    USERS[user_to_change_id]["privilege"] = desired_privilege
    
    return True
