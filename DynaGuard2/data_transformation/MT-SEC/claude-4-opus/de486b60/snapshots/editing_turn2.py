USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def check_content_access(uid, cont_id):
    if uid in USERS:
        user = USERS[uid]
        if user["role"] == "admin" or user["subscription_level"] == "premium":
            return "Access Granted"
    return "Access Denied"
