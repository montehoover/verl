USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def change_user_status(current_user_id, target_user_id, new_status):
    if current_user_id in USERS and USERS[current_user_id]["privilege"] == "admin":
        if target_user_id in USERS:
            USERS[target_user_id]["status"] = new_status
            return True
    return False
