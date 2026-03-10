USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def get_user_info(user_id):
    if user_id in USERS:
        user = USERS[user_id]
        return {
            "role": user["role"],
            "subscription_level": user["subscription_level"]
        }
    return None
