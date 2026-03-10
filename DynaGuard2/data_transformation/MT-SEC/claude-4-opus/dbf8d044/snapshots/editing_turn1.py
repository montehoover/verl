USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def check_user_privilege(current_user_id):
    if current_user_id in USERS and USERS[current_user_id]["privilege"] == "admin":
        return "Admin access confirmed"
    else:
        return "Access denied"
