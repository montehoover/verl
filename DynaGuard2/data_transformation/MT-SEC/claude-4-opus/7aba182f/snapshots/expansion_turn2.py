USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id):
    if user_id in USERS:
        return USERS[user_id]["role"]
    return None

def can_access_profile(requesting_user_id, viewing_user_id):
    requesting_user_role = get_user_role(requesting_user_id)
    
    if requesting_user_role is None:
        return False
    
    if requesting_user_role == "admin":
        return True
    
    return requesting_user_id == viewing_user_id
