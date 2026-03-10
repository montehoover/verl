USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_valid_privilege(privilege: str) -> bool:
    valid_privileges = ["normal", "moderator", "admin"]
    return privilege in valid_privileges

def user_exists(user_id: int, users: dict) -> bool:
    return user_id in users

def user_has_admin_privilege(user_id: int, users: dict) -> bool:
    return users.get(user_id, {}).get("privilege") == "admin"

def update_user_privilege(user_id: int, new_privilege: str, users: dict) -> None:
    users[user_id]["privilege"] = new_privilege

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    # Check if current user exists and has admin privileges
    if not user_exists(current_user_id, USERS):
        return False
    
    if not user_has_admin_privilege(current_user_id, USERS):
        return False
    
    # Check if target user exists
    if not user_exists(target_user_id, USERS):
        return False
    
    # Check if new privilege is valid
    if not is_valid_privilege(new_privilege):
        return False
    
    # Modify the privilege
    update_user_privilege(target_user_id, new_privilege, USERS)
    return True
