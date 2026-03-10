USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id):
    if user_id in USERS:
        return USERS[user_id]["privilege"] == "admin"
    return False

def can_modify_privilege(acting_user_id, affected_user_id):
    return is_admin(acting_user_id)

def update_privileges(acting_user_id, affected_user_id, new_role):
    if not can_modify_privilege(acting_user_id, affected_user_id):
        return False
    
    if affected_user_id not in USERS:
        return False
    
    role_to_privilege = {
        "admin": "admin",
        "moderator": "moderator",
        "user": "normal"
    }
    
    if new_role not in role_to_privilege:
        return False
    
    USERS[affected_user_id]["role"] = new_role
    USERS[affected_user_id]["privilege"] = role_to_privilege[new_role]
    
    return True
