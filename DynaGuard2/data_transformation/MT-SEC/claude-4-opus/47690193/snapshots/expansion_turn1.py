USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def check_user_role(user_id):
    if user_id in USERS:
        return USERS[user_id]["role"]
    return "unknown"
