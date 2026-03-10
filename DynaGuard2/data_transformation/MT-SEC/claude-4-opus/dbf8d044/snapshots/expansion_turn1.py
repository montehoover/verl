USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def get_user_info(user_id):
    if user_id in USERS:
        user = USERS[user_id]
        return {
            "role": user["role"],
            "privilege": user["privilege"]
        }
    return None
