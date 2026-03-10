USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

def get_user_role(user_id):
    if user_id in USER_ROLES:
        return USER_ROLES[user_id]["role"]
    return None

def can_access_invoice(user_attempt_id, cust_id):
    role = get_user_role(user_attempt_id)
    if role == "admin":
        return True
    elif role == "customer" and user_attempt_id == cust_id:
        return True
    return False
