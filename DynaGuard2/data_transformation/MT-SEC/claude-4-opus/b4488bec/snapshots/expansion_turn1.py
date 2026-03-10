USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

def is_admin_or_customer(user_id):
    if user_id in USER_ROLES:
        role = USER_ROLES[user_id]["role"]
        return role in ["admin", "customer"]
    return False
