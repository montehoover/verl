USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def check_user_role(uid):
    if uid in USERS and USERS[uid]["role"] == "admin":
        return "User is admin"
    else:
        return "User is not admin"
